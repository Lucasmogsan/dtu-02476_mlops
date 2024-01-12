import pytest
from tests import _PATH_DATA
import torch
import os.path

# Skip test if data files are not found
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason='Data files not found')
def test_data():
    # Assert length of dataset loaded
    data_folder = _PATH_DATA
    data_name = 'rice'
    train_images = torch.load(
        data_folder + 'processed/' + data_name + '/processed_train_images.pt',
    )
    train_labels = torch.load(
        data_folder + 'processed/' + data_name + '/processed_train_labels.pt',
    )
    trainset = torch.utils.data.TensorDataset(train_images, train_labels)

    test_images = torch.load(
        data_folder + 'raw/' + data_name + '/test_images.pt',
    )
    test_labels = torch.load(
        data_folder + 'raw/' + data_name + '/test_target.pt',
    )
    testset = torch.utils.data.TensorDataset(test_images, test_labels)

    assert len(trainset) == 25000, 'Trainset should have 25000 datapoints'
    assert len(testset) == 5000, 'Testset should have 5000 datapoints'

    # Assert each datapoint has shape 28x28 for a linear network
    for i in range(len(trainset)):
        # images : 1,28,28
        # labels: 1
        # dataset: (images, labels) = ([1,28,28], 1)
        assert trainset[i][0].shape == (28,28), 'Train image should be 28x28'

    for i in range(len(testset)):
        assert testset[i][0].shape == (28,28), 'Test image should be 28x28'

    # Assert that all labels are represented
    assert len(set(train_labels.tolist())) == 10, 'Train labels should have 10 classes'
    assert len(set(test_labels.tolist())) == 10, 'Test labels should have 10 classes'
