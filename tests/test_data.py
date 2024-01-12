import pytest
from tests import _PATH_DATA
import torch
import os.path


# Assert length of dataset loaded
data_folder = _PATH_DATA + "processed/"

n_files = 5
n_train_per_file = 400
n_test_per_file = 100

# Load pt files
train_dataset = []
test_dataset = []
for i in range(n_files):
    train_dataset.append(torch.load(data_folder + f"train_data_{i}.pt"))
    test_dataset.append(torch.load(data_folder + f"test_data_{i}.pt"))
train_dataset = torch.utils.data.ConcatDataset(train_dataset)
test_dataset = torch.utils.data.ConcatDataset(test_dataset)


# Skip test if data files are not found
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_length():
    expected_train = n_files * n_train_per_file
    expected_test = n_files * n_test_per_file
    assert len(train_dataset) == expected_train, "Trainset should have {expected_train} datapoints"
    assert len(test_dataset) == expected_test, "Testset should have {expected_test} datapoints"


def test_data_shape():
    # # Assert each processed tensor has shape 1x224x224
    for i in range(len(train_dataset)):
        #     # images : 1,224,224
        #     # labels: 1
        #     # dataset: (images, labels) = ([1,1,224,224], 1)
        assert train_dataset[i][0].shape == (1, 224, 224), "Train image should be 1,224,224"

    for i in range(len(test_dataset)):
        assert test_dataset[i][0].shape == (1, 224, 224), "Test image should be 28x28"


def test_data_labels():
    # Assert that all labels are represented
    train_labels = [label.item() for _, label in train_dataset]
    train_labels = set(train_labels)
    test_labels = [label.item() for _, label in test_dataset]
    test_labels = set(test_labels)
    assert len(train_labels) == 5, "Train labels should have 5 classes"
    assert len(test_labels) == 5, "Test labels should have 5 classes"
