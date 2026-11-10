# PoreTimeNet

PoreTimeNet segments sequences of three-dimensional images using a Swin Transformer spatial encoder, temporal attention and temporal decoder fusion. This framework includes data loading, training with sparse temporal annotations, validation, checkpoint saving and resumption, and sliding-window prediction.

## Installation

Use Python 3.10 or later and install PyTorch 2.6.0 for your CPU or CUDA environment. From the repository directory:

```bash
pip install .
```

Dependencies are specified in `pyproject.toml`. No datasets or pretrained weights are downloaded.

## Try the complete workflow

Generate small synthetic sequences, train a compact model for two epochs, and predict a separate sequence:

```bash
poretimenet-demo --output demo_data
poretimenet-train --config demo_data/config.json --manifest demo_data/manifest.json --output runs/demo --device cpu
poretimenet-predict --checkpoint runs/demo/best.pt --manifest demo_data/manifest.json --output predictions/demo --device cpu
```

The synthetic example checks the workflow; its small network, data and training schedule are not the paper's experimental configuration. Output directories must be new or empty, and existing predictions are protected from accidental replacement.

## Prepare your data

Each image file contains one three-dimensional grayscale image. Supported formats are NumPy (`.npy`), NIfTI (`.nii` or `.nii.gz`) and three-dimensional TIFF (`.tif` or `.tiff`). Images within each sequence must be registered to the same voxel grid and have identical dimensions. NIfTI affines must also agree. Array axes are used in their stored order; the framework does not register, reorient or resample images.

List images in chronological order in a JSON manifest. Paths are resolved relative to the manifest file. The keys under `labels` specify the zero-based image indices that have reference segmentations:

```json
{
  "training": [{
    "id": "train_sample",
    "images": ["train/image_0.npy", "train/image_1.npy", "train/image_2.npy"],
    "labels": {"0": "train/segmentation_0.npy", "2": "train/segmentation_2.npy"}
  }],
  "validation": [{
    "id": "validation_sample",
    "images": ["validation/image_0.npy", "validation/image_1.npy"],
    "labels": {"1": "validation/segmentation_1.npy"}
  }],
  "prediction": [{
    "id": "new_sample",
    "images": ["new/image_0.npy", "new/image_1.npy", "new/image_2.npy"]
  }]
}
```

A reference segmentation must have the same shape and geometry as its image. Class values are integers from `0` to `out_channels - 1`. Class `0` is an ordinary class. Use `data.ignore_index` (default `-1`) for unannotated voxels within a reference segmentation. Missing time-step annotations are omitted from `labels`; all images remain available as temporal input. Keep the same class meaning across every sequence.

Use separate files for training and validation, such as separately exported spatial regions or different samples. The workflow rejects shared training/validation file paths; users must also ensure the underlying regions do not overlap. Validation is optional. Prediction needs no reference segmentations.

## Configure and train

Copy `examples/config.json` and edit its settings, then run:

```bash
poretimenet-train --config config.json --manifest manifest.json --output runs/model --device cuda:0
```

The configuration is JSON; unknown options are rejected. The main settings are:

| Setting | Meaning |
| --- | --- |
| `model.out_channels` | Number of material classes |
| `model.temporal_size` | Odd number of images in each input window |
| `model.feature_size`, `model.depths` | Network width and encoder block counts |
| `data.patch_size` | Three spatial dimensions of each training/prediction patch |
| `data.temporal_strides` | Spacings in positions in the image list, not seconds |
| `data.samples_per_epoch` | Number of randomly sampled training patches per epoch |
| `data.normalization` | `none`, fixed-range scaling, or per-image `zscore` |
| `data.cache_size` | Maximum number of loaded volumes retained by each worker |
| `training.epochs`, `training.learning_rate` | Training duration and initial learning rate |
| `training.precision` | `float32`, or CUDA `float16`/`bfloat16` |
| `training.validation_interval` | Number of epochs between validation runs |
| `loss.class_weights` | Optional positive cross-entropy weights, one per class |

The workflow currently accepts one grayscale channel and runs on CPU or one GPU. The network defaults are `feature_size=48`, `depths=[2,2,2,2]` and `temporal_size=5`. Spatial patch dimensions must be divisible by 32; the deepest feature map needs more than one spatial voxel. For example, `[64,64,64]` and `[32,32,64]` are valid, but `[32,32,32]` is too small for default instance normalisation. Whole image dimensions need not be multiples of 32.

Fixed-range normalisation clips values to the specified `lower` and `upper` bounds and maps them to `[0,1]`. Choose these bounds for your data and use the same transform throughout the sequence. `zscore` instead uses each image's own mean and standard deviation. The chosen transform is stored in the checkpoint and reused during prediction; it is never applied to reference segmentations.

Training first samples a temporal stride uniformly, then a window with at least one available annotation. Reference segmentations can occur at different positions in that window. At sequence boundaries, the nearest endpoint image supplies missing temporal context; its annotation is not duplicated into padded positions. All images and annotations in a window receive the same spatial crop and spatial flips. Optional temporal reversal also reverses annotation positions. Augmentation switches are in `training.augmentation`.

Dice and cross-entropy losses use only annotated voxels. Losses are evaluated at each supervised temporal position and weighted by its number of active batch samples. Empty annotation slots are skipped; entirely unannotated training patches are retried and rejected if no valid patch can be found. Loss arithmetic uses float32 under mixed precision. The optimisers are AdamW, Adam or SGD, with optional linear warmup followed by cosine learning-rate decay.

## Checkpoints and validation

A training run writes:

- `last.pt`: model, optimiser, scheduler, mixed-precision scaler and random state after each completed epoch.
- `best.pt`: the checkpoint with the highest validation mean Dice score, when validation is provided.
- `config.json`: the complete resolved configuration, with no source data paths.
- `history.jsonl`: training loss, learning rate and validation scores by epoch.

Validation predicts each annotated image using the central output of its temporal window. The first configured temporal stride is used. Scores are computed on valid voxels for every class and reference image; a class absent from both prediction and reference is omitted for that image. The checkpoint score is the mean of these class-image Dice scores. Per-class averages and the number of evaluated images are also recorded.

To resume an interrupted run, use `last.pt` in the original output directory, keep `best.pt` if validation was used, and keep the original configuration and manifest:

```bash
poretimenet-train --config runs/model/config.json --manifest manifest.json --output runs/model --resume runs/model/last.pt --device cuda:0
```

Resumption starts at the next epoch. Changing the configuration during resumption is rejected. Keep the dataset and class mapping unchanged as well. Checkpoints are loaded with PyTorch's `weights_only=True` option.

## Predict your own sequences

```bash
poretimenet-predict --checkpoint runs/model/best.pt --manifest manifest.json --output predictions/model --device cuda:0 --precision bfloat16 --sw-batch-size 2
```

Use `last.pt` when training without validation. Set `--stride` to select the spacing between input images; it counts positions in the manifest. A centred window is constructed for every image, including the first and last images.

Each spatial patch is converted to class probabilities before Gaussian blending of overlapping patches. Only the central temporal output is accumulated, reducing memory use. This is equivalent to blending every temporal output and then selecting the centre. Validation and prediction use the same probability-blending procedure. `--overlap` defaults to `0.5` and `--sw-batch-size` controls patch batching.

CPU stitching is the default, so full images and accumulated predictions can remain in host memory while a GPU processes patches. Use `--stitch-device cuda:0` to keep them on a GPU when memory permits. CUDA mixed precision is optional and depends on hardware support. Large images still require enough host memory for a temporal window and the output probabilities; this implementation does not provide out-of-core or multi-GPU inference.

Predictions are saved as `output/<sequence-id>/0000_prediction.<original-format>`, one file per input image. NIfTI outputs retain the original geometry and spatial units; free-text header fields and extensions are not copied. The integer class IDs match the training targets. `predictions.json` lists relative output filenames and time indices. Existing predictions require an explicit `--overwrite`; input files are never overwritten.

## Python interface

```python
from poretimenet import PoreTimeNet
from poretimenet.inference import predict_window

model = PoreTimeNet(in_channels=1, out_channels=3, temporal_size=5)
```

The network takes tensors of shape `(batch, channels, time, depth, height, width)` and returns class logits with shape `(batch, classes, time, depth, height, width)`. The time dimension must match `temporal_size`. Load compatible weights before using predictions for analysis.

## Tests

```bash
pip install '.[test]'
python -m pytest
```

The tests cover sparse supervision, temporal padding, consistent image/annotation transforms, file geometry, sliding-window aggregation and output protection. The synthetic workflow above additionally exercises training, checkpoint creation and prediction with the real network.

## Scope and licensing

The network implementation is retained from the research model. The surrounding single-device workflow provides configurable sampling, normalisation, validation and file handling for new datasets. It does not bundle experimental images, trained weights, dataset-specific training schedules or cluster job scripts, and the example configuration is not a complete reproduction of the paper's experiments. No remote experiment-tracking service is used, and checkpoints do not store input paths, usernames or host information.

PoreTimeNet contributions are licensed under [CC BY-NC 4.0](LICENSE). Portions derived from MONAI retain the [Apache License 2.0](LICENSES/Apache-2.0.txt). See [NOTICE](NOTICE) for attribution and the scope of each license.
