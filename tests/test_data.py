import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from poretimenet.data import (
    PatchDataset, VolumeStore, load_window, read_manifest, read_volume,
    window_indices, write_prediction,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def save(self, name, array):
        path = self.root / name
        np.save(path, array, allow_pickle=False)
        return path

    def config(self, temporal_size=3, patch=(4, 4, 4), **data):
        return {
            'model': {'in_channels': 1, 'out_channels': 3, 'temporal_size': temporal_size},
            'data': {'patch_size': list(patch), 'samples_per_epoch': 4, 'temporal_strides': [1],
                     'normalization': {'mode': 'none'}, 'ignore_index': -1, **data},
            'training': {'seed': 42, 'augmentation': {'spatial_flip': False, 'temporal_reverse': False}},
        }

    def test_manifest_relative_paths_sparse_labels_and_validation(self):
        self.save('image.npy', np.zeros((4, 4, 4), dtype=np.float32))
        self.save('label.npy', np.ones((4, 4, 4), dtype=np.uint8))
        path = self.root / 'dataset.json'
        sequence = {'id': 'sample_1', 'images': ['image.npy'], 'labels': {'0': 'label.npy'}}
        path.write_text(json.dumps({'training': [sequence]}))
        loaded = read_manifest(path, 'training')
        self.assertEqual(loaded[0]['images'], [self.root / 'image.npy'])
        self.assertEqual(loaded[0]['labels'], {0: self.root / 'label.npy'})
        self.assertEqual(read_manifest(path, 'validation'), [])
        for invalid in [
            {'id': '../escape', 'images': ['image.npy']},
            {'id': 'sample', 'images': ['missing.npy']},
            {'id': 'sample', 'images': ['image.npy'], 'labels': {'1': 'label.npy'}},
            {'id': 'sample', 'images': ['image.npy'], 'labels': {'0.5': 'label.npy'}},
        ]:
            path.write_text(json.dumps({'training': [invalid]}))
            with self.assertRaises((ValueError, FileNotFoundError)):
                read_manifest(path, 'training')
        path.write_text(json.dumps({'training': [sequence, sequence]}))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_manifest(path, 'training')

    def test_volume_normalisation_and_label_validation(self):
        values = np.linspace(-2, 12, 64, dtype=np.float32).reshape(4, 4, 4)
        path = self.save('image.npy', values)
        fixed = VolumeStore({'mode': 'fixed', 'lower': 0, 'upper': 10})
        image, _ = fixed.get(path)
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_allclose(image, np.clip(values, 0, 10) / 10)
        zscore, _ = VolumeStore({'mode': 'zscore'}).get(path)
        self.assertAlmostEqual(float(zscore.mean()), 0, places=6)
        self.assertAlmostEqual(float(zscore.std()), 1, places=6)
        constant = self.save('constant.npy', np.ones((4, 4, 4)))
        self.assertTrue(np.all(VolumeStore({'mode': 'zscore'}).get(constant)[0] == 0))
        label = self.save('labels.npy', np.full((4, 4, 4), 2, dtype=np.uint8))
        labels, _ = fixed.get(label, label=True)
        self.assertEqual(labels.dtype, np.int64)
        self.assertTrue(np.all(labels == 2))
        with self.assertRaisesRegex(ValueError, 'integers'):
            fixed.get(path, label=True)
        invalid = self.save('nan.npy', np.full((4, 4, 4), np.nan))
        with self.assertRaises(ValueError):
            fixed.get(invalid)
        with self.assertRaises(ValueError):
            VolumeStore({'mode': 'fixed', 'lower': 1, 'upper': 1})

    def test_window_padding_does_not_duplicate_supervision(self):
        image = self.save('image.npy', np.full((4, 4, 4), 7, dtype=np.float32))
        label = self.save('label.npy', np.ones((4, 4, 4), dtype=np.uint8))
        sequence = {'id': 'sample', 'images': [image], 'labels': {0: label}}
        self.assertEqual(window_indices(0, 5, 1, 1), [0, 0, 0, 0, 0])
        window = load_window(sequence, 0, 5, 1, VolumeStore())
        self.assertEqual(tuple(window.shape), (1, 5, 4, 4, 4))
        sample = PatchDataset([sequence], self.config(temporal_size=5))[0]
        self.assertTrue(torch.all(sample['image'] == 7))
        target = sample['target']
        self.assertTrue(torch.all(target[2] == 1))
        self.assertTrue(torch.all(target[[0, 1, 3, 4]] == -1))

    def test_sparse_slots_track_real_images(self):
        images = [self.save(f'image_{i}.npy', np.full((4, 4, 4), i, np.float32)) for i in range(3)]
        labels = {i: self.save(f'label_{i}.npy', np.full((4, 4, 4), i, np.uint8)) for i in (0, 2)}
        sequence = {'id': 'sample', 'images': images, 'labels': labels}
        dataset = PatchDataset([sequence], self.config())
        for index in range(len(dataset)):
            sample = dataset[index]
            image, target = sample['image'][0], sample['target']
            valid = target != -1
            self.assertTrue(torch.any(valid))
            self.assertTrue(torch.equal(image[valid].long(), target[valid]))
            self.assertFalse(torch.any(target == 1))

    def test_spatial_and_temporal_transforms_remain_aligned(self):
        values = np.indices((6, 7, 8)).sum(axis=0) % 3
        image = self.save('image.npy', values.astype(np.float32))
        label = self.save('label.npy', values.astype(np.int64))
        sequence = {'id': 'sample', 'images': [image] * 5, 'labels': {i: label for i in range(5)}}
        config = self.config()
        config['training']['augmentation'] = {'spatial_flip': True, 'temporal_reverse': True}
        dataset = PatchDataset([sequence], config)
        for epoch in range(3):
            dataset.set_epoch(epoch)
            for index in range(len(dataset)):
                first, repeated = dataset[index], dataset[index]
                self.assertTrue(torch.equal(first['image'], repeated['image']))
                self.assertTrue(torch.equal(first['target'], repeated['target']))
                valid = first['target'] != -1
                self.assertTrue(torch.equal(first['image'][0][valid].long(), first['target'][valid]))

    def test_small_images_and_all_ignored_patches(self):
        image = self.save('image.npy', np.ones((2, 3, 4), dtype=np.float32))
        label = self.save('label.npy', np.full((2, 3, 4), 2, dtype=np.int64))
        sequence = {'id': 'sample', 'images': [image], 'labels': {0: label}}
        sample = PatchDataset([sequence], self.config(temporal_size=1))[0]
        self.assertEqual(tuple(sample['image'].shape), (1, 1, 4, 4, 4))
        self.assertTrue(torch.all(sample['image'] == 1))
        self.assertEqual(int((sample['target'] != -1).sum()), 24)
        ignored = self.save('ignored.npy', np.full((2, 3, 4), -1, dtype=np.int64))
        sequence['labels'] = {0: ignored}
        with self.assertRaisesRegex(ValueError, 'No patch with valid supervision'):
            PatchDataset([sequence], self.config(temporal_size=1, max_sampling_attempts=2))[0]
        bad = self.save('bad.npy', np.full((2, 3, 4), 7, dtype=np.int64))
        sequence['labels'] = {0: bad}
        with self.assertRaisesRegex(ValueError, 'class indices'):
            PatchDataset([sequence], self.config(temporal_size=1))[0]

    def test_nifti_geometry_and_output_roundtrip(self):
        image_path = self.root / 'image.nii.gz'
        affine = np.diag([0.1, 0.2, 0.3, 1.0])
        affine[:3, 3] = [4, 5, 6]
        image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), affine)
        image.header.set_xyzt_units('mm')
        image.header['descrip'] = b'private source description'
        image.header['aux_file'] = b'private source file'
        image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b'private metadata'))
        image.set_qform(affine, code=1)
        image.set_sform(affine, code=2)
        nib.save(image, image_path)
        values = (np.indices((4, 5, 6)).sum(axis=0) % 3).astype(np.int64)
        for extension in ('npy', 'tif', 'nii.gz'):
            output = self.root / f'prediction.{extension}'
            write_prediction(output, values, image_path)
            array, metadata = read_volume(output)
            np.testing.assert_array_equal(array, values)
            self.assertEqual(array.dtype, np.uint8)
            if extension == 'nii.gz':
                np.testing.assert_allclose(metadata['affine'], affine)
                self.assertEqual(metadata['header'].get_xyzt_units()[0], 'mm')
                self.assertEqual(int(metadata['header']['qform_code']), 1)
                self.assertEqual(int(metadata['header']['sform_code']), 2)
                self.assertEqual(bytes(metadata['header']['descrip']).strip(b'\x00'), b'')
                self.assertEqual(bytes(metadata['header']['aux_file']).strip(b'\x00'), b'')
                self.assertEqual(len(metadata['header'].extensions), 0)
        mismatch = self.root / 'mismatch.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.float32), np.eye(4)), mismatch)
        sequence = {'id': 'sample', 'images': [image_path, mismatch], 'labels': {0: self.root / 'prediction.nii.gz'}}
        with self.assertRaisesRegex(ValueError, 'affine'):
            load_window(sequence, 0, 3, 1, VolumeStore())
        with self.assertRaisesRegex(ValueError, 'affine'):
            PatchDataset([sequence], self.config())

    def test_shape_and_output_integer_checks(self):
        image = self.save('image.npy', np.zeros((4, 4, 4), np.float32))
        label = self.save('label.npy', np.zeros((3, 4, 4), np.int64))
        with self.assertRaisesRegex(ValueError, 'same shape'):
            PatchDataset([{'id': 'sample', 'images': [image], 'labels': {0: label}}], self.config())
        with self.assertRaisesRegex(ValueError, 'integer'):
            write_prediction(self.root / 'output.npy', np.full((4, 4, 4), 0.5), image)
        image_2d = self.save('two_dimensional.npy', np.zeros((4, 4)))
        with self.assertRaisesRegex(ValueError, 'three-dimensional'):
            read_volume(image_2d)


if __name__ == '__main__':
    unittest.main()
