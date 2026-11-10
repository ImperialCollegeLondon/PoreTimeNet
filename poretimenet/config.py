import copy
import json
import math
from pathlib import Path

DEFAULT_CONFIG = {
    'model': {'in_channels': 1, 'out_channels': 3, 'temporal_size': 5,
              'feature_size': 48, 'depths': [2, 2, 2, 2], 'num_heads': [3, 6, 12, 24],
              'window_size': 7, 'patch_size': 2, 'spatial_dims': 3,
              'use_checkpoint': False, 'use_v2': False, 'drop_rate': 0.0,
              'attn_drop_rate': 0.0, 'dropout_path_rate': 0.0,
              'use_encoder_temporal_block': True, 'use_decoder_temporal_fusion': True},
    'data': {'patch_size': [64, 64, 64], 'temporal_strides': [1],
             'samples_per_epoch': 100, 'cache_size': 2, 'max_sampling_attempts': 32,
             'normalization': {'mode': 'none', 'lower': 0.0, 'upper': 1.0},
             'ignore_index': -1},
    'training': {'epochs': 100, 'batch_size': 1, 'learning_rate': 1e-4,
                 'weight_decay': 1e-5, 'optimizer': 'adamw', 'warmup_epochs': 0,
                 'min_learning_rate': 1e-7, 'validation_interval': 1,
                 'num_workers': 0, 'seed': 42, 'precision': 'float32',
                 'gradient_clip': 1.0,
                 'augmentation': {'spatial_flip': True, 'temporal_reverse': True}},
    'loss': {'dice_weight': 1.0, 'ce_weight': 1.0, 'label_smoothing': 0.05,
             'class_weights': None},
    'inference': {'overlap': 0.5, 'sw_batch_size': 2, 'stitch_device': 'cpu'},
}


def _merge(default, supplied, prefix=''):
    if not isinstance(supplied, dict):
        raise ValueError(f'{prefix or "Configuration"} must be a JSON object')
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        if key not in default:
            raise ValueError(f'Unknown configuration option: {prefix}{key}')
        result[key] = _merge(default[key], value, f'{prefix}{key}.') if isinstance(default[key], dict) else value
    return result


def validate_config(config):
    c = _merge(DEFAULT_CONFIG, config)
    m, d, t, loss = c['model'], c['data'], c['training'], c['loss']
    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    def finite(value, name, minimum=0.0, strict=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        if value < minimum or (strict and value == minimum):
            raise ValueError(f'{name} is outside its permitted range')
    if m['in_channels'] != 1 or m['spatial_dims'] != 3 or m['patch_size'] != 2:
        raise ValueError('The file-based workflow requires one grayscale channel, three spatial dimensions and model.patch_size=2')
    integer(m['out_channels'], 'model.out_channels', 2)
    if m['out_channels'] > 65535:
        raise ValueError('At most 65535 output classes are supported')
    integer(m['temporal_size'], 'model.temporal_size')
    if m['temporal_size'] % 2 != 1:
        raise ValueError('model.temporal_size must be odd for central-image prediction')
    integer(m['feature_size'], 'model.feature_size')
    for key in ('use_checkpoint', 'use_v2', 'use_encoder_temporal_block', 'use_decoder_temporal_fusion'):
        if not isinstance(m[key], bool):
            raise ValueError(f'model.{key} must be true or false')
    windows = m['window_size'] if isinstance(m['window_size'], list) else [m['window_size']]
    if isinstance(m['window_size'], list) and len(windows) != 3:
        raise ValueError('model.window_size must be an integer or three integers')
    for v in windows:
        integer(v, 'model.window_size')
    for key in ('drop_rate', 'attn_drop_rate', 'dropout_path_rate'):
        finite(m[key], key)
        if m[key] > 1:
            raise ValueError(f'model.{key} must be in [0,1]')
    if m['feature_size'] % 12:
        raise ValueError('model.feature_size must be divisible by 12')
    for key in ('depths', 'num_heads'):
        if not isinstance(m[key], list) or len(m[key]) != 4:
            raise ValueError(f'model.{key} requires four values')
        for v in m[key]:
            integer(v, f'model.{key}')
    for i, heads in enumerate(m['num_heads']):
        if m['feature_size'] * (2**i) % heads:
            raise ValueError('Each encoder feature size must be divisible by its attention head count')
    if not isinstance(d['patch_size'], list) or len(d['patch_size']) != 3:
        raise ValueError('data.patch_size requires three spatial dimensions')
    for v in d['patch_size']:
        integer(v, 'data.patch_size')
        if v % 32:
            raise ValueError('Every patch dimension must be divisible by 32')
    if math.prod(v // 32 for v in d['patch_size']) <= 1:
        raise ValueError('The deepest encoder stage needs more than one spatial voxel; use a larger patch')
    if not isinstance(d['temporal_strides'], list) or not d['temporal_strides']:
        raise ValueError('data.temporal_strides requires at least one value')
    for v in d['temporal_strides']:
        integer(v, 'data.temporal_strides')
    if len(set(d['temporal_strides'])) != len(d['temporal_strides']):
        raise ValueError('data.temporal_strides must be unique')
    integer(d['samples_per_epoch'], 'data.samples_per_epoch')
    integer(d['max_sampling_attempts'], 'data.max_sampling_attempts')
    integer(d['cache_size'], 'data.cache_size', 0)
    if isinstance(d['ignore_index'], bool) or not isinstance(d['ignore_index'], int) or 0 <= d['ignore_index'] < m['out_channels']:
        raise ValueError('data.ignore_index must be an integer outside the class range')
    norm = d['normalization']
    if norm['mode'] not in ('none', 'fixed', 'zscore'):
        raise ValueError('normalization.mode must be none, fixed or zscore')
    if norm['mode'] == 'fixed' and not (math.isfinite(norm['lower']) and math.isfinite(norm['upper']) and norm['upper'] > norm['lower']):
        raise ValueError('Fixed normalisation requires finite lower < upper')
    for key in ('epochs', 'batch_size', 'validation_interval'):
        integer(t[key], f'training.{key}')
    for key in ('num_workers', 'warmup_epochs', 'seed'):
        integer(t[key], f'training.{key}', 0)
    if t['warmup_epochs'] >= t['epochs']:
        raise ValueError('warmup_epochs must be less than epochs')
    if t['optimizer'] not in ('adamw', 'adam', 'sgd'):
        raise ValueError('optimizer must be adamw, adam or sgd')
    if t['precision'] not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('precision must be float32, float16 or bfloat16')
    finite(t['learning_rate'], 'learning_rate', strict=True)
    for key in ('weight_decay', 'min_learning_rate', 'gradient_clip'):
        finite(t[key], key)
    if t['min_learning_rate'] > t['learning_rate']:
        raise ValueError('min_learning_rate must not exceed learning_rate')
    for key in ('dice_weight', 'ce_weight', 'label_smoothing'):
        finite(loss[key], key)
    if loss['dice_weight'] + loss['ce_weight'] == 0 or loss['label_smoothing'] > 1:
        raise ValueError('Loss weights must not both be zero; label_smoothing must be in [0,1]')
    weights = loss['class_weights']
    if weights is not None:
        if not isinstance(weights, list) or len(weights) != m['out_channels']:
            raise ValueError('class_weights must have one entry per class')
        for w in weights:
            finite(w, 'class_weights', strict=True)
    for v in t['augmentation'].values():
        if not isinstance(v, bool):
            raise ValueError('Augmentation switches must be true or false')
    inf = c['inference']
    finite(inf['overlap'], 'inference.overlap')
    if inf['overlap'] >= 1:
        raise ValueError('overlap must be less than one')
    integer(inf['sw_batch_size'], 'inference.sw_batch_size')
    if inf['stitch_device'] not in ('cpu', 'cuda'):
        raise ValueError('stitch_device must be cpu or cuda')
    return c


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))
