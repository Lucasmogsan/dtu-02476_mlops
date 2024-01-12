import torch
import pytest
from tests import _MODEL_PATH


# Assert given input with shape X, model returns output with shape Y
def test_model_forward():
    x = torch.randn((1, 224, 224))
    x = x.unsqueeze(0)  # (1, 1, 224, 224), added a dimension for batch size
    model = torch.load(_MODEL_PATH + "model_latest1.pt")
    y = model.forward(x)
    assert y.shape == (1, 5), "Model output should have shape (1,5)"


def test_model_raises():
    """Check that model makes the correct raises"""

    # Missing batch dimension
    with pytest.raises(
        ValueError,
        match="not enough values to unpack",
    ):
        x = torch.randn((1, 224, 224))
        model = torch.load(_MODEL_PATH + "model_latest1.pt")
        _ = model.forward(x)

    # # Wrong image size
    # with pytest.raises(
    #     AssertionError,
    #     match=f"Input height ," "doesn't match model",
    # ):
    #     x = torch.randn((1, 1, 200, 200))
    #     model = torch.load(_MODEL_PATH+"model_latest1.pt")
    #     _ = model.forward(x)
