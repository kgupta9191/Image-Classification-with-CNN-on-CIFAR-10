import importlib
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.import_module("src.transfer_cnn")


def _tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))


def _tiny_loader(num_samples=16, batch_size=4):
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_ci_and_pytest_config_allow_test_discovery():
    workflow = (ROOT / ".github" / "workflows" / "python-ci.yml").read_text(encoding="utf-8")
    cfg = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "pytest -v" in workflow
    assert "testpaths = tests" in cfg


def test_required_dependencies_importable():
    module_name_overrides = {
        "pillow": "PIL",
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
        "pyyaml": "yaml",
    }
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    deps = []
    for line in requirements:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        deps.append(re.split(r"[<>=!]", stripped)[0].strip())
    for dep in deps:
        import_name = module_name_overrides.get(dep.lower(), dep.replace("-", "_"))
        importlib.import_module(import_name)


def test_transfer_module_import_has_no_training_side_effect_globals():
    assert not hasattr(MODULE, "train_loader")
    assert not hasattr(MODULE, "val_loader")
    assert not hasattr(MODULE, "model")


def test_train_one_epoch_returns_valid_metric_ranges():
    model = _tiny_model()
    loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss, acc = MODULE.train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0


def test_evaluate_returns_valid_metric_ranges():
    model = _tiny_model()
    loader = _tiny_loader()
    criterion = nn.CrossEntropyLoss()

    loss, acc = MODULE.evaluate(model, loader, criterion, torch.device("cpu"))

    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0


def test_output_tensor_shape_matches_class_count():
    model = _tiny_model()
    x = torch.randn(5, 3, 32, 32)
    out = model(x)
    assert out.shape == (5, 10)


def test_optimizer_step_updates_model_parameters():
    model = _tiny_model()
    loader = _tiny_loader(num_samples=8, batch_size=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]

    MODULE.train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    after = list(model.parameters())

    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, after))


def test_scheduler_step_changes_learning_rate():
    model = _tiny_model()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_before = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    lr_after = optimizer.param_groups[0]["lr"]
    assert lr_after < lr_before


def test_device_handling_cpu_gpu_fallback_works():
    device = MODULE.get_default_device()
    assert device.type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_train_and_test_transforms_output_shape_and_finite_values():
    train_transform, test_transform = MODULE.get_transforms(target_image_size=224)
    image = Image.fromarray(np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8))

    train_tensor = train_transform(image)
    test_tensor = test_transform(image)

    assert train_tensor.shape == (3, 224, 224)
    assert test_tensor.shape == (3, 224, 224)
    assert torch.isfinite(train_tensor).all()
    assert torch.isfinite(test_tensor).all()


def test_dataloader_batch_shape_and_label_format_are_correct():
    x_train = torch.randn(20, 3, 224, 224)
    y_train = torch.randint(0, 10, (20,), dtype=torch.long)
    x_test = torch.randn(6, 3, 224, 224)
    y_test = torch.randint(0, 10, (6,), dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader, val_loader, test_loader, split_train_dataset = MODULE.create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        test_transform=None,
        loader_batch_size=4,
    )
    assert len(split_train_dataset) > 0

    for loader in (train_loader, val_loader, test_loader):
        images, labels = next(iter(loader))
        assert images.shape[1:] == (3, 224, 224)
        assert labels.ndim == 1
        assert labels.dtype == torch.long


def test_mini_end_to_end_one_epoch_completes_and_returns_metrics():
    model = _tiny_model()
    train_loader = _tiny_loader(num_samples=12, batch_size=4)
    val_loader = _tiny_loader(num_samples=8, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    train_loss, train_acc = MODULE.train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = MODULE.evaluate(model, val_loader, criterion, device)

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= val_acc <= 1.0
