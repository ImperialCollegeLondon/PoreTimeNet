import pytest
import torch
from monai.losses import DiceLoss
from torch.nn import functional as F

from poretimenet.config import DEFAULT_CONFIG, validate_config
from poretimenet.loss import SparseSegmentationLoss


def test_unannotated_frames_and_voxels_have_no_loss_gradient():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 3, 3, 2, 2, 2, generator=generator, requires_grad=True)
    target = torch.full((2, 3, 2, 2, 2), -1, dtype=torch.long)
    target[0, 1] = 1
    target[1, 2] = 2
    target[0, 1, 0, 0, 0] = -1
    loss, _ = SparseSegmentationLoss(3)(logits, target)
    loss.backward()
    ignored = (target == -1).unsqueeze(1).expand_as(logits)
    assert torch.count_nonzero(logits.grad[ignored]) == 0
    assert torch.count_nonzero(logits.grad[~ignored]) > 0
    assert torch.isfinite(logits.grad).all()


def test_all_ignored_batch_is_rejected():
    logits = torch.zeros(1, 3, 3, 2, 2, 2)
    target = torch.full((1, 3, 2, 2, 2), -1, dtype=torch.long)
    with pytest.raises(ValueError, match="no supervised"):
        SparseSegmentationLoss(3)(logits, target)


def test_fully_ignored_slot_is_skipped():
    generator = torch.Generator().manual_seed(2)
    logits = torch.randn(1, 3, 3, 2, 2, 2, generator=generator)
    target = torch.zeros(1, 3, 2, 2, 2, dtype=torch.long)
    target[:, 1] = -1
    target[:, 2] = 2
    criterion = SparseSegmentationLoss(3)
    full_loss, full_components = criterion(logits, target)
    selected_loss, selected_components = criterion(logits[:, :, [0, 2]], target[:, [0, 2]])
    torch.testing.assert_close(full_loss, selected_loss)
    for key in full_components:
        torch.testing.assert_close(full_components[key], selected_components[key])


def test_supervised_time_slots_use_active_sample_weights():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 3, 3, 2, 2, 2, generator=generator)
    target = torch.full((2, 3, 2, 2, 2), -1, dtype=torch.long)
    target[0, 0] = 0
    target[0, 1] = 1
    target[1, 1] = 2
    target[1, 2] = 0
    class_weights = [0.8, 1.2, 2.0]
    criterion = SparseSegmentationLoss(
        3, dice_weight=0.7, ce_weight=1.3, label_smoothing=0.05,
        class_weights=class_weights,
    )
    actual, components = criterion(logits, target)
    dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=True,
                    batch=True, smooth_nr=1e-5, smooth_dr=1e-5)
    slots = (([0], 0), ([0, 1], 1), ([1], 2))
    expected_dice = torch.tensor(0.0)
    expected_ce = torch.tensor(0.0)
    for active, time in slots:
        prediction = logits[active, :, time]
        truth = target[active, time]
        expected_dice += len(active) * dice(prediction, truth.unsqueeze(1))
        expected_ce += len(active) * F.cross_entropy(
            prediction, truth, weight=torch.tensor(class_weights), label_smoothing=0.05,
        )
    expected_dice /= 4
    expected_ce /= 4
    torch.testing.assert_close(components['dice_loss'], expected_dice)
    torch.testing.assert_close(components['ce_loss'], expected_ce)
    torch.testing.assert_close(actual, 0.7 * expected_dice + 1.3 * expected_ce)


def test_out_of_range_target_is_rejected():
    logits = torch.zeros(1, 3, 1, 2, 2, 2)
    target = torch.full((1, 1, 2, 2, 2), 3, dtype=torch.long)
    with pytest.raises(ValueError, match="class indices"):
        SparseSegmentationLoss(3)(logits, target)


@pytest.mark.parametrize('supplied', [
    {'model': {'temporal_size': 4}},
    {'model': {'num_heads': [5, 6, 12, 24]}},
    {'model': {'unsupported_option': True}},
    {'data': {'patch_size': [32, 32, 32]}},
    {'data': {'patch_size': [64, 64, 63]}},
    {'data': {'temporal_strides': [1, 1]}},
    {'data': {'ignore_index': 0}},
    {'data': {'normalization': {'mode': 'fixed', 'lower': 1, 'upper': 0}}},
    {'loss': {'class_weights': [1, 2]}},
    {'loss': {'dice_weight': 0, 'ce_weight': 0}},
    {'training': {'epochs': 0}},
    {'training': {'warmup_epochs': 100}},
    {'training': {'augmentation': {'spatial_flip': 'false'}}},
    {'training': {'learning_rate': float('nan')}},
    {'inference': {'overlap': 1}},
])
def test_invalid_configuration_is_rejected(supplied):
    with pytest.raises(ValueError):
        validate_config(supplied)


def test_configuration_defaults_are_not_mutated():
    config = validate_config({'model': {'temporal_size': 3},
                              'data': {'patch_size': [32, 32, 64]}})
    config['data']['patch_size'][0] = 128
    assert DEFAULT_CONFIG['data']['patch_size'] == [64, 64, 64]
    assert DEFAULT_CONFIG['model']['temporal_size'] == 5
