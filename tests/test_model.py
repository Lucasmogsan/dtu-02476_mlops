from tests import _PROJECT_ROOT, _PATH_DATA
from mlops_group8.models.model import MyNeuralNet
import torch
import pytest

# Assert given input with shape X, model returns output with shape Y
def test_model():
    x = torch.randn((1,784))
    model = MyNeuralNet()
    y = model.forward(x)
    # assert y.shape == torch.Size([10])
    assert y.shape == (1,10), 'Model output should have shape (1,10)'

def test_model_raises():
    '''Check that model makes the correct raises'''
    with pytest.raises(
        ValueError, match='Expected input tensor to have 2 dimensions, '
        'got tensor with shape {x.shape}',
    ):
        x = torch.randn((1,784,1))
        model = MyNeuralNet()
        y = model.forward(x)

    with pytest.raises(
        ValueError, match='Expected input tensor to have 784 features, '
        'got tensor with shape {x.shape}',
    ):
        x = torch.randn((1,800))
        model = MyNeuralNet()
        y = model.forward(x)

# Parametrize test
@pytest.mark.parametrize('test_input,expected', [('3+5', 8), ('2+4', 6), ('6*9', 54)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
