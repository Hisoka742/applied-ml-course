import numpy as np
import pytest
import torch

from eurosat_ssl.augmentations import AUGMENTATION_LEVEL_NAMES, build_augmentation, tta_views
from eurosat_ssl.data import BatchPreprocessor, ImageTensorDataset, SplitConfig, make_splits
from eurosat_ssl.models import CONVNEXT_TINY, DEIT_SMALL, ArchitectureSpec, create_model, layer_groups
from eurosat_ssl.pseudo_labeling import select_pseudo_labels
from eurosat_ssl.training import Stage, TrainConfig, Trainer, make_loader

# A tiny ConvNeXt keeps the training test fast on CPU.
CONVNEXT_ATTO = ArchitectureSpec("ConvNeXt-atto", "convnext_atto")


def random_images(n_images: int, size: int = 64) -> torch.Tensor:
    return torch.randint(0, 256, (n_images, 3, size, size), dtype=torch.uint8)


@pytest.mark.parametrize("spec", [CONVNEXT_TINY, DEIT_SMALL])
def test_layer_groups_cover_every_parameter_once(spec: ArchitectureSpec) -> None:
    model = create_model(spec, n_classes=10, image_size=64, pretrained=False)
    groups = layer_groups(model)
    names = [name for group in groups for name, _ in group]
    assert sorted(names) == sorted(name for name, _ in model.named_parameters())
    head_names = {name for name, _ in model.named_parameters() if name.startswith("head")}
    assert head_names <= {name for name, _ in groups[0]}
    assert len(groups) == 5


@pytest.mark.parametrize("level", range(len(AUGMENTATION_LEVEL_NAMES)))
def test_augmentation_keeps_shape_and_dtype(level: int) -> None:
    image = random_images(1)[0]
    augmented = build_augmentation(level)(image)
    assert augmented.shape == image.shape
    assert augmented.dtype == torch.uint8


def test_augmentation_level_zero_is_identity() -> None:
    image = random_images(1)[0]
    assert torch.equal(build_augmentation(0)(image), image)


def test_tta_views_keep_shape() -> None:
    images = random_images(2)
    assert all(view.shape == images.shape for view in tta_views(images))


def test_splits_are_disjoint_and_stratified() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    splits = make_splits(labels, SplitConfig(test_share=0.2, val_share=0.05, labeled_share=0.1, seed=0))
    parts = [splits.labeled, splits.unlabeled, splits.val, splits.test]
    all_indices = np.concatenate(parts)
    assert len(all_indices) == len(set(all_indices)) == len(labels)
    assert len(splits.test) == 400
    assert len(splits.val) == 100
    assert np.bincount(labels.numpy()[splits.labeled]).tolist() == [15] * 10


def test_batch_preprocessor_resizes_and_normalizes() -> None:
    preprocessor = BatchPreprocessor(96, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    output = preprocessor(torch.full((2, 3, 64, 64), 255, dtype=torch.uint8))
    assert output.shape == (2, 3, 96, 96)
    assert torch.allclose(output, torch.ones_like(output))


def test_select_pseudo_labels_respects_threshold_and_class_caps() -> None:
    probabilities = torch.tensor(
        [
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.60, 0.40],
            [0.05, 0.95],
        ]
    )
    pseudo = select_pseudo_labels(probabilities, threshold=0.9, class_prior=np.array([0.5, 0.5]), slack=1.0)
    # 4 confident predictions, so every class may get at most ceil(0.5 * 4) = 2 pseudo-labels.
    assert pseudo.indices.tolist() == [0, 1, 4]
    assert pseudo.labels.tolist() == [0, 0, 1]


def test_staged_training_unfreezes_groups_and_logs_history() -> None:
    torch.manual_seed(0)
    n_classes = 3
    images, labels = random_images(48, size=32), torch.arange(48) % n_classes
    config = TrainConfig(batch_size=16, num_workers=0)
    model = create_model(CONVNEXT_ATTO, n_classes, image_size=32, pretrained=False)
    trainer = Trainer(model, BatchPreprocessor(32, (0.5,) * 3, (0.5,) * 3), config, torch.device("cpu"))
    val_loader = make_loader(ImageTensorDataset(images, labels), config, shuffle=False)

    def make_train_set(level: int) -> ImageTensorDataset:
        return ImageTensorDataset(images, labels, build_augmentation(level))

    stages = [
        Stage("head", n_unfrozen_groups=1, augmentation_level=0, base_lr=1e-3, max_epochs=2),
        Stage("all", n_unfrozen_groups=None, augmentation_level=2, base_lr=1e-4, max_epochs=1),
    ]
    trainer.fit(stages, make_train_set, val_loader)

    history = trainer.history_frame()
    assert history["stage"].tolist() == ["head", "head", "all"]
    assert history["unfrozen_groups"].tolist() == [1, 1, 5]
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert trainer.predict_proba(val_loader, use_tta=True).shape == (48, n_classes)


def test_discriminative_learning_rates_decay_away_from_head() -> None:
    model = create_model(CONVNEXT_ATTO, n_classes=3, image_size=32, pretrained=False)
    trainer = Trainer(
        model, BatchPreprocessor(32, (0.5,) * 3, (0.5,) * 3), TrainConfig(), torch.device("cpu")
    )
    stage = Stage(
        "all", n_unfrozen_groups=None, augmentation_level=0, base_lr=1e-3, max_epochs=1, lr_decay=0.5
    )
    optimizer = trainer._make_optimizer(stage, n_unfrozen=trainer._set_trainable(None))
    learning_rates = sorted({group["lr"] for group in optimizer.param_groups}, reverse=True)
    assert learning_rates == pytest.approx([1e-3 * 0.5**i for i in range(5)])
