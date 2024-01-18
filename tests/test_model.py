import torch
import pytest
from tests import _MODEL_PATH


model_name = "model_latest.pt"


# Assert given input with shape X, model returns output with shape Y
def test_model_forward():
    x = torch.randn((1, 224, 224))
    x = x.unsqueeze(0)  # (1, 1, 224, 224), added a dimension for batch size
    model = torch.load(_MODEL_PATH + model_name)
    y = model.forward(x)
    assert y.shape == (1, 5), "Model output should have shape (1,5)"


# Assert ValueErrors from the model
def test_model_raises():
    # Missing batch dimension
    with pytest.raises(
        ValueError,
        match="not enough values to unpack",
    ):
        x = torch.randn((1, 224, 224))
        model = torch.load(_MODEL_PATH + model_name)
        _ = model.forward(x)

    # Wrong image input size
    input_size = 200
    with pytest.raises(
        AssertionError,
        match=f"Input height \\({input_size}\\) doesn't match model \\(224\\)",
    ):
        x = torch.randn((1, 1, input_size, input_size))
        model = torch.load(_MODEL_PATH + model_name)
        _ = model.forward(x)
