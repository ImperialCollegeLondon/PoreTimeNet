import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import tempfile

import torch
from monai.inferers import sliding_window_inference

from .data import VolumeStore, load_window, read_manifest, write_prediction
from .model import PoreTimeNet


def _device(value, name):
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be a CPU or CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"{name} requests CUDA, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"{name} refers to an unavailable CUDA device")
    return device


def _precision_context(device, precision):
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be float32, float16 or bfloat16")
    if precision == "float32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("float16 and bfloat16 inference require a CUDA device")
    if precision == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("The selected CUDA device does not support bfloat16")
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def predict_window(
    model,
    images,
    patch_size,
    overlap=0.5,
    sw_batch_size=2,
    device="cpu",
    stitch_device="cpu",
    precision="float32",
):
    """Return centre-image class probabilities as a CPU tensor (C, D, H, W)."""
    device = _device(device, "device")
    stitch_device = _device(stitch_device, "stitch_device")
    _precision_context(device, precision)
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim == 5:
        images = images.unsqueeze(0)
    if images.ndim != 6 or images.shape[0] != 1 or images.shape[1] != 1:
        raise ValueError("images must have shape (1, T, D, H, W) or (1, 1, T, D, H, W)")
    if any(size < 1 for size in images.shape):
        raise ValueError("Image dimensions must be positive")
    time_steps = images.shape[2]
    if time_steps % 2 != 1:
        raise ValueError("Centre-image inference requires an odd number of time steps")
    if getattr(model, "temporal_size", time_steps) != time_steps:
        raise ValueError("The input time length does not match model.temporal_size")
    if not isinstance(sw_batch_size, int) or isinstance(sw_batch_size, bool) or sw_batch_size < 1:
        raise ValueError("sw_batch_size must be a positive integer")
    if not isinstance(overlap, (float, int)) or not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive)")
    patch_size = tuple(patch_size)
    divisor = int(getattr(model, "patch_size", 2)) * 16
    if len(patch_size) != 3 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < divisor or size % divisor
        for size in patch_size
    ):
        raise ValueError(f"patch_size must contain three positive multiples of {divisor}")
    has_instance_norm = any(isinstance(module, torch.nn.InstanceNorm3d) for module in model.modules())
    if has_instance_norm and math.prod(size // divisor for size in patch_size) <= 1:
        raise ValueError("patch_size is too small for instance normalisation at the deepest encoder stage")
    if not torch.isfinite(images).all():
        raise ValueError("Input images contain non-finite values")

    images = images.detach().to(device=stitch_device, dtype=torch.float32)
    image_5d = images.reshape(1, time_steps, *images.shape[-3:])
    center = time_steps // 2
    model = model.to(device)
    was_training = model.training
    model.eval()

    def predictor(patch):
        batch = patch.shape[0]
        patch = patch.reshape(batch, 1, time_steps, *patch.shape[-3:])
        with _precision_context(device, precision):
            logits = model(patch)
        if logits.ndim != 6 or logits.shape[0] != batch or logits.shape[2] != time_steps:
            raise ValueError("The model must return logits with shape (batch, classes, time, D, H, W)")
        return torch.softmax(logits, dim=1)[:, :, center]

    try:
        with torch.inference_mode():
            probabilities = sliding_window_inference(
                inputs=image_5d,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=float(overlap),
                mode="gaussian",
                sw_device=device,
                device=stitch_device,
                progress=False,
            )[0].float().cpu()
    finally:
        model.train(was_training)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("The model produced non-finite probabilities; try float32 inference")
    return probabilities


def _output_suffix(path):
    name = Path(path).name.lower()
    for suffix in (".nii.gz", ".nii", ".npy", ".tiff", ".tif"):
        if name.endswith(suffix):
            return suffix
    raise ValueError("Prediction images must use NPY, NIfTI or TIFF files")


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(content, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_prediction(
    checkpoint,
    manifest,
    output_dir,
    split="prediction",
    device="cpu",
    precision="float32",
    stride=1,
    overlap=0.5,
    sw_batch_size=2,
    stitch_device="cpu",
    overwrite=False,
):
    """Segment every image in a manifest split using centred temporal windows."""
    device = _device(device, "device")
    stitch_device = _device(stitch_device, "stitch_device")
    _precision_context(device, precision)
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    checkpoint = Path(checkpoint).resolve()
    manifest = Path(manifest).resolve()
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(stored, dict) or not isinstance(stored.get("config"), dict):
        raise ValueError("The checkpoint must contain config and model_state_dict")
    state = stored.get("model_state_dict")
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("model_state_dict must map parameter names to tensors")
    if stored.get("format_version", 1) != 1:
        raise ValueError("The checkpoint format version is not supported")
    config = stored["config"]
    try:
        json.dumps(config, allow_nan=False)
        model_config = config["model"]
        data_config = config["data"]
        patch_size = data_config["patch_size"]
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("The checkpoint configuration is incomplete or is not JSON-compatible") from error
    if not isinstance(model_config, dict) or not isinstance(data_config, dict):
        raise ValueError("The checkpoint model and data configurations must be objects")
    if model_config.get("in_channels", 1) != 1:
        raise ValueError("The prediction workflow expects single-channel image sequences")
    length = model_config.get("temporal_size", 5)
    if not isinstance(length, int) or length < 1 or length % 2 != 1:
        raise ValueError("temporal_size must be a positive odd integer")
    sequences = read_manifest(manifest, split)
    if not sequences:
        raise ValueError(f"The manifest split {split!r} contains no sequences")
    output_dir = Path(output_dir).resolve()
    source_paths = {checkpoint, manifest}
    for sequence in sequences:
        source_paths.update(Path(path).resolve() for path in sequence["images"])
        source_paths.update(Path(path).resolve() for path in sequence.get("labels", {}).values())
    planned = []
    seen_paths = set()
    for sequence in sequences:
        identifier = sequence["id"]
        if not isinstance(identifier, str) or identifier in {"", ".", ".."} or Path(identifier).name != identifier or "\\" in identifier:
            raise ValueError("Sequence identifiers must be simple directory names")
        for center, source in enumerate(sequence["images"]):
            relative = Path(identifier) / f"{center:04d}_prediction{_output_suffix(source)}"
            destination = (output_dir / relative).resolve()
            if not destination.is_relative_to(output_dir) or destination in source_paths:
                raise ValueError("An output path would leave the output directory or overwrite an input")
            if destination in seen_paths:
                raise ValueError("Multiple predictions would use the same output path")
            if destination.exists() and not overwrite:
                raise FileExistsError(f"Prediction already exists: {relative}; use overwrite=True to replace it")
            seen_paths.add(destination)
            planned.append((sequence, center, source, destination, relative))
    summary_path = (output_dir / "predictions.json").resolve()
    if not summary_path.is_relative_to(output_dir) or summary_path in source_paths:
        raise ValueError("The prediction summary would overwrite an input or leave the output directory")
    if summary_path.exists() and not overwrite:
        raise FileExistsError("predictions.json already exists; use overwrite=True to replace it")
    model = PoreTimeNet(**model_config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    store = VolumeStore(data_config.get("normalization", {"mode": "none"}), cache_size=data_config.get("cache_size", 2))
    records = []
    for sequence, center, source, destination, relative in planned:
        images = load_window(sequence, center, length, stride, store)
        probabilities = predict_window(
            model, images, patch_size, overlap=overlap, sw_batch_size=sw_batch_size,
            device=device, stitch_device=stitch_device, precision=precision,
        )
        prediction = probabilities.argmax(dim=0).numpy()
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_prediction(destination, prediction, source)
        records.append({"sequence": sequence["id"], "time_index": center, "file": relative.as_posix()})
        print(f"Saved {relative.as_posix()}", flush=True)
    summary = {"format_version": 1, "predictions": records}
    _write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Segment image sequences with a trained PoreTimeNet model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="prediction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--stitch-device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        run_prediction(
            args.checkpoint, args.manifest, args.output, split=args.split, device=args.device,
            precision=args.precision, stride=args.stride, overlap=args.overlap,
            sw_batch_size=args.sw_batch_size, stitch_device=args.stitch_device, overwrite=args.overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
