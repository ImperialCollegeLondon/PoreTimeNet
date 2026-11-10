import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, validate_config
from .data import PatchDataset, VolumeStore, load_window, read_manifest
from .inference import predict_window
from .loss import SparseSegmentationLoss
from .model import PoreTimeNet


def resolve_device(value):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Choose cpu, cuda or a CUDA device such as cuda:0')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available')
    return device


def _autocast(device, precision):
    if precision == 'float32':
        return nullcontext()
    dtype = torch.float16 if precision == 'float16' else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_optimizer(model, config):
    t = config['training']
    options = {'lr': t['learning_rate'], 'weight_decay': t['weight_decay']}
    if t['optimizer'] == 'adamw':
        return torch.optim.AdamW(model.parameters(), **options)
    if t['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(), **options)
    return torch.optim.SGD(model.parameters(), momentum=0.9, **options)


def make_scheduler(optimizer, config):
    t = config['training']
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, t['epochs'] - t['warmup_epochs']), eta_min=t['min_learning_rate'])
    if t['warmup_epochs'] == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6,
                                               end_factor=1.0, total_iters=t['warmup_epochs'])
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[t['warmup_epochs']])


def _save(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _dice(prediction, target, classes, ignore_index):
    valid = target != ignore_index
    if not valid.any():
        raise ValueError('A validation segmentation contains no valid voxels')
    if ((target[valid] < 0) | (target[valid] >= classes)).any():
        raise ValueError('Validation class indices are outside the configured class range')
    values = []
    for c in range(classes):
        p, y = (prediction == c) & valid, (target == c) & valid
        denominator = int(p.sum()) + int(y.sum())
        values.append(2.0 * int((p & y).sum()) / denominator if denominator else None)
    return values


@torch.inference_mode()
def evaluate(model, sequences, config, device):
    model.eval()
    d, m, t, inf = config['data'], config['model'], config['training'], config['inference']
    store = VolumeStore(d['normalization'], d['cache_size'])
    rows = []
    for sequence in sequences:
        for center, label_path in sequence['labels'].items():
            images = load_window(sequence, center, m['temporal_size'], d['temporal_strides'][0], store)
            truth, label_meta = store.get(label_path, label=True)
            _, image_meta = store.get(sequence['images'][center])
            if label_meta.get('affine') is not None and image_meta.get('affine') is not None:
                if not np.allclose(label_meta['affine'], image_meta['affine']):
                    raise ValueError('Validation images and segmentations have different NIfTI geometry')
            probabilities = predict_window(model, images, d['patch_size'], overlap=inf['overlap'],
                                           sw_batch_size=inf['sw_batch_size'], device=device,
                                           stitch_device=inf['stitch_device'], precision=t['precision'])
            prediction = probabilities.argmax(dim=0).numpy()
            if prediction.shape != truth.shape:
                raise ValueError('Validation image and segmentation shapes differ')
            rows.append(_dice(prediction, truth, m['out_channels'], d['ignore_index']))
    if not rows:
        raise ValueError('Validation sequences require at least one reference segmentation')
    per_class = []
    for c in range(m['out_channels']):
        values = [row[c] for row in rows if row[c] is not None]
        per_class.append(sum(values) / len(values) if values else None)
    values = [v for row in rows for v in row if v is not None]
    return {'mean_dice': sum(values) / len(values), 'per_class_dice': per_class,
            'evaluated_images': len(rows)}


def run_training(config_path, manifest_path, output_dir, device='auto', resume=None, threads=4):
    config = load_config(config_path)
    device = resolve_device(device)
    t, d, m = config['training'], config['data'], config['model']
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    if t['precision'] != 'float32' and device.type != 'cuda':
        raise ValueError('float16 and bfloat16 training require CUDA; use float32 on CPU')
    if t['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support bfloat16')
    sequences = read_manifest(manifest_path, 'training')
    validation = read_manifest(manifest_path, 'validation')
    if not sequences:
        raise ValueError('The manifest has no training sequences')
    training_files = {p for s in sequences for p in [*s['images'], *s['labels'].values()]}
    validation_files = {p for s in validation for p in [*s['images'], *s['labels'].values()]}
    if training_files & validation_files:
        raise ValueError('Training and validation must use separate files; export separate spatial regions if needed')
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError('Output directory is not empty; choose a new directory or use --resume')
    if resume is not None and Path(resume).resolve() != (output / 'last.pt').resolve():
        raise ValueError('Resume from last.pt in the original output directory')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(t['seed'])
    np.random.seed(t['seed'])
    torch.manual_seed(t['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(t['seed'])
    dataset = PatchDataset(sequences, config)
    loader = DataLoader(dataset, batch_size=t['batch_size'], num_workers=t['num_workers'],
                        shuffle=False, pin_memory=device.type == 'cuda', persistent_workers=False)
    model = PoreTimeNet(**m).to(device)
    criterion = SparseSegmentationLoss(m['out_channels'], ignore_index=d['ignore_index'], **config['loss']).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and t['precision'] == 'float16')
    start_epoch, best_metric, best_epoch = 0, None, None
    if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1:
            raise ValueError('Unsupported checkpoint format')
        if validate_config(checkpoint['config']) != config:
            raise ValueError('Resume requires the same training configuration; use the saved config.json')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch, best_metric = checkpoint['epoch'], checkpoint['best_metric']
        best_epoch = checkpoint.get('best_epoch')
        if best_metric is not None:
            best_path = output / 'best.pt'
            if not best_path.is_file():
                raise ValueError('The original best.pt is required to resume this validated run')
            best_saved = torch.load(best_path, map_location='meta', weights_only=True)
            if best_saved['epoch'] != best_epoch or best_epoch > start_epoch or best_saved['best_metric'] != best_metric:
                raise ValueError('last.pt and best.pt must be from the same completed training run')
            del best_saved
        torch.set_rng_state(checkpoint['rng_state']['torch'])
        random.setstate(checkpoint['rng_state']['python'])
        if device.type == 'cuda' and checkpoint['rng_state']['cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
    if start_epoch >= t['epochs']:
        raise ValueError('This checkpoint has already completed the configured number of epochs')
    (output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    history_path = output / 'history.jsonl'
    if resume is not None and history_path.exists():
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row['epoch'] <= start_epoch]
        history_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    for epoch in range(start_epoch, t['epochs']):
        dataset.set_epoch(epoch)
        model.train()
        total_loss, total_samples = 0.0, 0
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, t['precision']):
                logits = model(images)
                loss, _ = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Training loss is not finite; check the input intensities and configuration')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if t['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=not scaler.is_enabled())
            if t['gradient_clip'] == 0 and not scaler.is_enabled():
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Training gradients are not finite')
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
            total_samples += len(images)
        row = {'epoch': epoch + 1, 'training_loss': total_loss / total_samples,
               'learning_rate': optimizer.param_groups[0]['lr']}
        improved = False
        if validation and ((epoch + 1) % t['validation_interval'] == 0 or epoch + 1 == t['epochs']):
            scores = evaluate(model, validation, config, device)
            row['validation'] = scores
            if best_metric is None or scores['mean_dice'] > best_metric:
                best_metric, improved = scores['mean_dice'], True
                best_epoch = epoch + 1
        scheduler.step()
        checkpoint = {
            'format_version': 1, 'config': config, 'epoch': epoch + 1,
            'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'best_metric': best_metric, 'best_epoch': best_epoch,
            'rng_state': {'torch': torch.get_rng_state(), 'python': random.getstate(),
                          'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []},
        }
        if improved:
            _save(output / 'best.pt', checkpoint)
        _save(output / 'last.pt', checkpoint)
        with history_path.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    return output / 'last.pt'


def main():
    parser = argparse.ArgumentParser(description='Train PoreTimeNet with complete or sparse temporal annotations.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    try:
        run_training(args.config, args.manifest, args.output, args.device, args.resume, args.threads)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
