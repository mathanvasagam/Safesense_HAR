import pytest
import torch

from safesense.model import HousseinBaseline


def _inputs(batch: int = 3) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(batch, 32, 92),
        torch.randn(batch, 32, 60),
        torch.randn(batch, 32, 6),
        torch.ones(batch, 32),
        torch.ones(batch, 32),
        torch.ones(batch, 32),
    )


def test_baseline_outputs_and_gate_weights() -> None:
    model = HousseinBaseline()
    logits, weights = model(*_inputs())
    assert logits.shape == (3, 27)
    assert weights.shape == (3, 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_masked_mean_ignores_invalid_frames_and_fully_masked_branch_is_finite() -> None:
    model = HousseinBaseline()
    sequence = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)
    pooled = model._masked_mean(sequence, mask)
    assert pooled[0, 0].item() == 1.0
    assert pooled[1, 0].item() == 0.0

    values = _inputs(2)
    depth_mask = torch.cat((torch.ones(1, 32), torch.zeros(1, 32)))
    logits, weights = model(*values[:3], depth_mask, values[4], values[5])
    assert torch.isfinite(logits).all()
    assert torch.isfinite(weights).all()


def test_baseline_backward_is_finite() -> None:
    model = HousseinBaseline()
    inputs = _inputs(2)
    logits, weights = model(*inputs)
    loss = logits.square().mean() + weights.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_baseline_module_dimensions() -> None:
    model = HousseinBaseline()
    assert model.depth_lstm.input_size == 92
    assert model.depth_lstm.hidden_size == 128
    assert model.skeleton_conv.in_channels == 60
    assert model.skeleton_conv.out_channels == 128
    assert model.skeleton_conv.kernel_size == (3,)
    assert isinstance(model.skeleton_norm, torch.nn.BatchNorm1d)
    assert model.skeleton_lstm.hidden_size == 128
    assert model.imu_conv.in_channels == 6
    assert model.imu_conv.out_channels == 128
    assert model.imu_conv.kernel_size == (3,)
    assert isinstance(model.imu_norm, torch.nn.BatchNorm1d)
    assert model.imu_lstm.hidden_size == 128
    assert model.gate[0].in_features == 384
    assert model.gate[0].out_features == 384
    assert isinstance(model.gate[1], torch.nn.Tanh)
    assert model.gate[-1].out_features == 3
    assert isinstance(model.classifier[0], torch.nn.BatchNorm1d)
    assert model.classifier[1].in_features == 384
    assert model.classifier[1].out_features == 128
    assert model.classifier[-1].out_features == 27


def test_conv_initialization_is_seeded_and_zero_bias() -> None:
    torch.manual_seed(7)
    first = HousseinBaseline()
    torch.manual_seed(7)
    second = HousseinBaseline()
    torch.testing.assert_close(first.skeleton_conv.weight, second.skeleton_conv.weight)
    torch.testing.assert_close(first.imu_conv.weight, second.imu_conv.weight)
    assert torch.all(first.skeleton_conv.bias == 0)
    assert torch.all(first.imu_conv.bias == 0)
    assert 0.01 < float(first.skeleton_conv.weight.detach().std()) < 0.09


def test_branch_dropout_receives_lstm_sequences() -> None:
    model = HousseinBaseline()
    seen_shapes: list[tuple[int, ...]] = []
    model.branch_dropout.register_forward_hook(lambda _module, inputs, _output: seen_shapes.append(tuple(inputs[0].shape)))
    model(*_inputs(2))
    assert len(seen_shapes) == 3
    assert all(shape == (2, 32, 128) for shape in seen_shapes)


def test_classifier_width_is_configurable() -> None:
    model = HousseinBaseline(classifier_width=1024)
    assert model.classifier[1].out_features == 1024
    assert model.classifier[-1].in_features == 1024


def test_baseline_rejects_wrong_shapes() -> None:
    model = HousseinBaseline()
    inputs = _inputs(2)
    with pytest.raises(ValueError, match="depth must have shape"):
        model(inputs[0][:, :31], *inputs[1:3])
    with pytest.raises(ValueError, match="skeleton_mask must have shape"):
        model(*inputs[:3], inputs[3], torch.ones(2, 31), inputs[5])
