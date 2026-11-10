import pytest
import torch
from monai.inferers import sliding_window_inference

from poretimenet.inference import predict_window


class ContextModel(torch.nn.Module):
    temporal_size = 3
    patch_size = 2

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, images):
        signal = images[:, 0]
        context = signal.mean(dim=1, keepdim=True)
        local = signal.mean(dim=(-3, -2, -1), keepdim=True)
        return torch.stack((self.scale * signal, context.expand_as(signal), -signal + local), dim=1)


def full_time_probabilities(model, images, patch_size, overlap=0.5, batch_size=2):
    time_steps = images.shape[1]
    flat = images.unsqueeze(0).reshape(1, time_steps, *images.shape[-3:])

    def predictor(patch):
        batch = patch.shape[0]
        logits = model(patch.reshape(batch, 1, time_steps, *patch.shape[-3:]))
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.reshape(batch, 3 * time_steps, *patch.shape[-3:])

    with torch.inference_mode():
        probabilities = sliding_window_inference(
            flat, patch_size, batch_size, predictor, overlap=overlap,
            mode="gaussian", sw_device="cpu", device="cpu",
        )
    return probabilities.reshape(1, 3, time_steps, *images.shape[-3:])[0, :, time_steps // 2]


@pytest.mark.parametrize("spatial_shape", [(40, 45, 38), (15, 22, 10)])
def test_center_matches_full_time_probability_aggregation(spatial_shape):
    model = ContextModel().eval()
    generator = torch.Generator().manual_seed(11)
    images = torch.randn((1, 3, *spatial_shape), generator=generator)
    expected = full_time_probabilities(model, images, (32, 32, 32), batch_size=3)
    observed = predict_window(model, images, (32, 32, 32), sw_batch_size=3)
    assert observed.shape == (3, *spatial_shape)
    assert observed.device.type == "cpu"
    assert observed.dtype == torch.float32
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(observed.sum(dim=0), torch.ones(spatial_shape), rtol=0, atol=2e-6)
    assert not observed.requires_grad


def test_six_dimensional_input_and_training_mode_restored():
    model = ContextModel().train()
    images = torch.randn(1, 3, 15, 17, 19)
    first = predict_window(model, images, (32, 32, 32))
    second = predict_window(model, images.unsqueeze(0), (32, 32, 32))
    assert model.training
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_cpu_rejects_cuda_precision(precision):
    with pytest.raises(ValueError, match="require a CUDA device"):
        predict_window(ContextModel(), torch.ones(1, 3, 8, 8, 8), (32, 32, 32), precision=precision)


def test_rejects_invalid_window_and_nonfinite_images():
    model = ContextModel()
    with pytest.raises(ValueError, match="odd"):
        predict_window(model, torch.ones(1, 4, 8, 8, 8), (32, 32, 32))
    with pytest.raises(ValueError, match="multiples"):
        predict_window(model, torch.ones(1, 3, 8, 8, 8), (31, 32, 32))
    with pytest.raises(ValueError, match="overlap"):
        predict_window(model, torch.ones(1, 3, 8, 8, 8), (32, 32, 32), overlap=1.0)
    images = torch.ones(1, 3, 8, 8, 8)
    images[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        predict_window(model, images, (32, 32, 32))


def test_prediction_output_summary_and_overwrite_guard(tmp_path, monkeypatch):
    import json
    import numpy as np
    import poretimenet.inference as inference

    config = {"model": {"in_channels": 1, "out_channels": 3, "temporal_size": 3},
              "data": {"patch_size": [32, 32, 32], "normalization": {"mode": "none"}}}
    checkpoint = tmp_path / "weights.pt"
    torch.save({"format_version": 1, "config": config, "model_state_dict": ContextModel().state_dict()}, checkpoint)
    manifest = tmp_path / "images.json"
    manifest.write_text("{}")
    images = []
    for index in range(2):
        path = tmp_path / f"image_{index}.npy"
        np.save(path, np.full((9, 10, 11), index, dtype=np.float32))
        images.append(path)
    sequence = {"id": "sequence", "images": images, "labels": {}}
    monkeypatch.setattr(inference, "read_manifest", lambda path, split: [sequence])
    monkeypatch.setattr(inference, "PoreTimeNet", lambda **kwargs: ContextModel())
    output = tmp_path / "output"
    summary = inference.run_prediction(checkpoint, manifest, output)
    assert len(summary["predictions"]) == 2
    for index in range(2):
        result = np.load(output / "sequence" / f"{index:04d}_prediction.npy")
        assert result.shape == (9, 10, 11)
        assert np.issubdtype(result.dtype, np.integer)
    text = (output / "predictions.json").read_text()
    assert str(tmp_path) not in text
    assert json.loads(text) == summary
    with pytest.raises(FileExistsError):
        inference.run_prediction(checkpoint, manifest, output)


def test_prediction_cannot_overwrite_source_even_with_overwrite_enabled(tmp_path, monkeypatch):
    import numpy as np
    import poretimenet.inference as inference

    output = tmp_path / "output"
    source = output / "sequence" / "0000_prediction.npy"
    source.parent.mkdir(parents=True)
    data = np.zeros((9, 10, 11), dtype=np.float32)
    np.save(source, data)
    checkpoint = tmp_path / "weights.pt"
    config = {"model": {"in_channels": 1, "out_channels": 3, "temporal_size": 3},
              "data": {"patch_size": [32, 32, 32], "normalization": {"mode": "none"}}}
    torch.save({"format_version": 1, "config": config, "model_state_dict": ContextModel().state_dict()}, checkpoint)
    manifest = tmp_path / "images.json"
    manifest.write_text("{}")
    monkeypatch.setattr(inference, "read_manifest", lambda path, split: [{"id": "sequence", "images": [source], "labels": {}}])
    monkeypatch.setattr(inference, "PoreTimeNet", lambda **kwargs: ContextModel())
    with pytest.raises(ValueError, match="overwrite an input"):
        inference.run_prediction(checkpoint, manifest, output, overwrite=True)
    np.testing.assert_array_equal(np.load(source), data)
