from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith('.nii') or name.endswith('.nii.gz'):
        return 'nifti'
    if name.endswith('.npy'):
        return 'npy'
    if name.endswith(('.tif', '.tiff')):
        return 'tiff'
    raise ValueError(f'Unsupported volume format: {path.name}')


def _check_shape(shape) -> tuple[int, int, int]:
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError(f'Expected a nonempty three-dimensional volume, got {shape}')
    return shape


def _geometry(path: Path) -> dict[str, Any]:
    kind = _format(path)
    if kind == 'nifti':
        import nibabel as nib
        image = nib.load(str(path))
        return {'format': kind, 'shape': _check_shape(image.shape), 'affine': np.array(image.affine, copy=True)}
    if kind == 'npy':
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        return {'format': kind, 'shape': _check_shape(array.shape)}
    import tifffile
    with tifffile.TiffFile(path) as image:
        return {'format': kind, 'shape': _check_shape(image.series[0].shape)}


def _same_geometry(reference: dict, candidate: dict) -> None:
    if tuple(reference['shape']) != tuple(candidate['shape']):
        raise ValueError('All images and segmentations in a sequence must have the same shape')
    if 'affine' in reference and 'affine' in candidate:
        if not np.allclose(reference['affine'], candidate['affine'], rtol=0.0, atol=1e-6):
            raise ValueError('All NIfTI volumes in a sequence must use the same affine')


def read_manifest(path, split: str) -> list[dict]:
    """Read sequences and resolve volume paths relative to a JSON manifest."""
    if split not in {'training', 'validation', 'prediction'}:
        raise ValueError('split must be training, validation or prediction')
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError('The manifest must be an object containing sequence groups')
    entries = manifest.get(split, [])
    if not isinstance(entries, list):
        raise ValueError(f'The {split} group must be a list')
    result, ids = [], set()

    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Volume paths must be nonempty strings')
        volume = Path(value)
        if not volume.is_absolute():
            volume = path.parent / volume
        volume = volume.resolve()
        if not volume.is_file():
            raise FileNotFoundError(volume)
        _format(volume)
        return volume

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each sequence must be an object')
        identifier = entry.get('id')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', identifier):
            raise ValueError('Sequence ids must be safe filenames containing letters, digits, dots, underscores or hyphens')
        if identifier in ids:
            raise ValueError(f'Duplicate sequence id: {identifier}')
        ids.add(identifier)
        images = entry.get('images')
        if not isinstance(images, list) or not images:
            raise ValueError(f'Sequence {identifier} needs a nonempty images list')
        images = [resolve(value) for value in images]
        labels = entry.get('labels', {})
        if not isinstance(labels, dict):
            raise ValueError('labels must map zero-based image indices to segmentation paths')
        resolved_labels = {}
        for key, value in labels.items():
            if not isinstance(key, str) or not re.fullmatch(r'0|[1-9][0-9]*', key):
                raise ValueError('Label indices must be nonnegative integer strings')
            index = int(key)
            if index >= len(images):
                raise ValueError(f'Label index {index} is outside sequence {identifier}')
            resolved_labels[index] = resolve(value)
        result.append({'id': identifier, 'images': images, 'labels': resolved_labels})
    return result


def read_volume(path) -> tuple[np.ndarray, dict]:
    """Read one 3D volume without changing its stored axis order."""
    path = Path(path)
    kind = _format(path)
    metadata = {'format': kind}
    if kind == 'npy':
        array = np.load(path, mmap_mode='r', allow_pickle=False)
    elif kind == 'nifti':
        import nibabel as nib
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
        metadata.update(affine=np.array(image.affine, copy=True), header=image.header.copy(), image_class=type(image))
    else:
        import tifffile
        array = tifffile.imread(path)
    metadata['shape'] = _check_shape(array.shape)
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)) or np.iscomplexobj(array):
        raise ValueError('Volumes must contain real numeric values')
    return array, metadata


class VolumeStore:
    """Load volumes with an LRU cache; normalisation applies only to images."""

    def __init__(self, normalization: dict | None = None, cache_size: int = 2):
        self.normalization = dict(normalization or {'mode': 'none'})
        self.mode = self.normalization.get('mode', 'none')
        if self.mode not in {'none', 'fixed', 'zscore'}:
            raise ValueError('normalization.mode must be none, fixed or zscore')
        if self.mode == 'fixed':
            self.lower = float(self.normalization['lower'])
            self.upper = float(self.normalization['upper'])
            if not np.isfinite([self.lower, self.upper]).all() or self.upper <= self.lower:
                raise ValueError('Fixed normalisation requires finite lower < upper')
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError('cache_size cannot be negative')
        self.cache = OrderedDict()

    def get(self, path, label: bool = False) -> tuple[np.ndarray, dict]:
        key = (Path(path).resolve(), bool(label))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        array, metadata = read_volume(key[0])
        if not np.isfinite(array).all():
            raise ValueError('Volumes must not contain NaN or infinite values')
        if label:
            if not np.issubdtype(array.dtype, np.integer) and not np.issubdtype(array.dtype, np.bool_):
                if not np.equal(array, np.floor(array)).all():
                    raise ValueError('Segmentation values must be integers')
            if int(array.min()) < np.iinfo(np.int64).min or int(array.max()) > np.iinfo(np.int64).max:
                raise ValueError('Segmentation values exceed the supported integer range')
            array = np.array(array, dtype=np.int64, copy=True)
        else:
            array = np.array(array, dtype=np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ValueError('Image values exceed the float32 range')
            if self.mode == 'fixed':
                np.clip(array, self.lower, self.upper, out=array)
                array -= self.lower
                array /= self.upper - self.lower
            elif self.mode == 'zscore':
                mean = float(np.mean(array, dtype=np.float64))
                std = float(np.std(array, dtype=np.float64))
                array -= mean
                if std > 0:
                    array /= std
                else:
                    array.fill(0)
            if not np.isfinite(array).all():
                raise ValueError('Normalisation produced non-finite values')
        array.setflags(write=False)
        result = array, metadata
        if self.cache_size:
            self.cache[key] = result
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result


def write_prediction(path, array, reference_path) -> None:
    """Save integer predictions, preserving NIfTI geometry when available."""
    path = Path(path)
    reference = _geometry(Path(reference_path))
    array = np.asarray(array)
    if _check_shape(array.shape) != tuple(reference['shape']):
        raise ValueError('Prediction shape must match its reference image')
    if not np.isfinite(array).all() or np.iscomplexobj(array) or not np.equal(array, np.floor(array)).all() or array.min() < 0:
        raise ValueError('Predictions must contain nonnegative integer class indices')
    maximum = int(array.max())
    if maximum > np.iinfo(np.uint32).max:
        raise ValueError('Predicted class indices exceed uint32')
    dtype = np.uint8 if maximum <= 255 else np.uint16 if maximum <= 65535 else np.uint32
    array = array.astype(dtype)
    kind = _format(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'npy':
        with path.open('wb') as handle:
            np.save(handle, array, allow_pickle=False)
    elif kind == 'tiff':
        import tifffile
        tifffile.imwrite(path, array, photometric='minisblack')
    else:
        import nibabel as nib
        if reference['format'] == 'nifti':
            image = nib.load(str(reference_path))
            output = type(image)(array, image.affine)
            output.header.set_data_dtype(dtype)
            output.header.set_zooms(image.header.get_zooms()[:3])
            output.header.set_xyzt_units(*image.header.get_xyzt_units())
            output.set_qform(*image.get_qform(coded=True))
            output.set_sform(*image.get_sform(coded=True))
        else:
            output = nib.Nifti1Image(array, np.eye(4))
        nib.save(output, str(path))


def window_indices(center: int, length: int, stride: int, count: int) -> list[int]:
    if length < 1 or stride < 1 or count < 1 or not 0 <= center < count:
        raise ValueError('Window length, stride and count must be positive, with a valid centre index')
    return [min(max(center + (i - length // 2) * stride, 0), count - 1) for i in range(length)]


def load_window(sequence: dict, center: int, length: int, stride: int, store: VolumeStore) -> torch.Tensor:
    """Load a temporal window as (1, time, depth, height, width)."""
    indices = window_indices(center, length, stride, len(sequence['images']))
    arrays, reference = [], None
    for index in indices:
        array, metadata = store.get(sequence['images'][index])
        if reference is None:
            reference = dict(metadata)
        else:
            _same_geometry(reference, metadata)
            if 'affine' not in reference and 'affine' in metadata:
                reference['affine'] = metadata['affine']
        arrays.append(array)
    return torch.from_numpy(np.stack(arrays)).unsqueeze(0)


class PatchDataset(Dataset):
    """Sample spatial patches with sparse supervision across a time window."""

    def __init__(self, sequences: list[dict], config: dict):
        self.sequences = sequences
        data = config.get('data', {})
        training = config.get('training', {})
        model = config.get('model', {})
        self.patch_size = _check_shape(data.get('patch_size', [64, 64, 64]))
        self.temporal_size = int(model.get('temporal_size', 5))
        self.num_classes = int(model.get('out_channels', 3))
        if int(model.get('in_channels', 1)) != 1:
            raise ValueError('The volume loader supports one greyscale channel per image')
        if self.temporal_size < 1 or self.num_classes < 1:
            raise ValueError('temporal_size and out_channels must be positive')
        self.strides = [int(value) for value in data.get('temporal_strides', [1])]
        if not self.strides or any(value <= 0 for value in self.strides) or len(set(self.strides)) != len(self.strides):
            raise ValueError('temporal_strides must contain distinct positive integers')
        self.samples_per_epoch = int(data.get('samples_per_epoch', 100))
        self.max_sampling_attempts = int(data.get('max_sampling_attempts', 32))
        self.seed = int(training.get('seed', 0))
        if self.samples_per_epoch < 1 or self.max_sampling_attempts < 1 or self.seed < 0:
            raise ValueError('Sample counts must be positive and seed nonnegative')
        self.epoch = 0
        self.ignore_index = int(data.get('ignore_index', -1))
        if 0 <= self.ignore_index < self.num_classes:
            raise ValueError('ignore_index must differ from every class index')
        augmentation = training.get('augmentation', {})
        self.spatial_flip = bool(augmentation.get('spatial_flip', True))
        self.temporal_reverse = bool(augmentation.get('temporal_reverse', True))
        self.store = VolumeStore(data.get('normalization'), data.get('cache_size', 2))
        self.geometry = []
        self.candidates = {stride: [] for stride in self.strides}
        for sequence_index, sequence in enumerate(sequences):
            if not sequence['images']:
                raise ValueError('Each sequence needs at least one image')
            reference = None
            paths = list(sequence['images']) + list(sequence.get('labels', {}).values())
            for path in paths:
                metadata = _geometry(Path(path))
                if reference is None:
                    reference = dict(metadata)
                else:
                    _same_geometry(reference, metadata)
                    if 'affine' not in reference and 'affine' in metadata:
                        reference['affine'] = metadata['affine']
            self.geometry.append(reference)
            count = len(sequence['images'])
            labels = sequence.get('labels', {})
            if any(not isinstance(index, int) or index < 0 or index >= count for index in labels):
                raise ValueError('Labels must use valid integer image indices')
            for stride in self.strides:
                for center in range(count):
                    raw = [center + (i - self.temporal_size // 2) * stride for i in range(self.temporal_size)]
                    if any(index in labels for index in raw):
                        self.candidates[stride].append((sequence_index, center))
        if not any(self.candidates.values()):
            raise ValueError('Training requires at least one labelled image')
        self.strides = [stride for stride in self.strides if self.candidates[stride]]

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError('epoch cannot be negative')
        self.epoch = int(epoch)

    def _crop(self, array: np.ndarray, starts, label: bool = False):
        slices = tuple(slice(start, start + width) for start, width in zip(starts, self.patch_size))
        cropped = array[slices]
        padding = [(0, width - size) for width, size in zip(self.patch_size, cropped.shape)]
        if label:
            return np.pad(cropped, padding, mode='constant', constant_values=self.ignore_index)
        return np.pad(cropped, padding, mode='edge')

    def __getitem__(self, index):
        if not 0 <= index < self.samples_per_epoch:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        for _ in range(self.max_sampling_attempts):
            stride = self.strides[int(rng.integers(len(self.strides)))]
            candidates = self.candidates[stride]
            sequence_index, center = candidates[int(rng.integers(len(candidates)))]
            sequence = self.sequences[sequence_index]
            count = len(sequence['images'])
            reference = self.geometry[sequence_index]
            starts = [int(rng.integers(max(size - width, 0) + 1)) for size, width in zip(reference['shape'], self.patch_size)]
            raw = [center + (i - self.temporal_size // 2) * stride for i in range(self.temporal_size)]
            clamped = window_indices(center, self.temporal_size, stride, count)
            images = []
            target = np.full((self.temporal_size, *self.patch_size), self.ignore_index, dtype=np.int64)
            for slot, frame in enumerate(clamped):
                image, metadata = self.store.get(sequence['images'][frame])
                _same_geometry(reference, metadata)
                images.append(self._crop(image, starts))
                source_index = raw[slot]
                if 0 <= source_index < count and source_index in sequence.get('labels', {}):
                    label, metadata = self.store.get(sequence['labels'][source_index], label=True)
                    _same_geometry(reference, metadata)
                    invalid = (label != self.ignore_index) & ((label < 0) | (label >= self.num_classes))
                    if np.any(invalid):
                        raise ValueError('Segmentation values must be class indices or ignore_index')
                    target[slot] = self._crop(label, starts, label=True)
            if not np.any(target != self.ignore_index):
                continue
            images = np.stack(images)
            if self.spatial_flip:
                for axis in (1, 2, 3):
                    if rng.random() < 0.5:
                        images = np.flip(images, axis=axis)
                        target = np.flip(target, axis=axis)
            if self.temporal_reverse and rng.random() < 0.5:
                images = np.flip(images, axis=0)
                target = np.flip(target, axis=0)
            return {'image': torch.from_numpy(np.ascontiguousarray(images)).unsqueeze(0), 'target': torch.from_numpy(np.ascontiguousarray(target))}
        raise ValueError('No patch with valid supervision was sampled; check labels, patch size and max_sampling_attempts')
