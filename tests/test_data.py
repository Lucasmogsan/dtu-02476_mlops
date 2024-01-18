import pytest
import torch
import os.path
from hydra import initialize, compose
from tests import _DATA_PATH, _PROJECT_NAME


processed_data_folder = _DATA_PATH + "processed/"
config_path = "../" + _PROJECT_NAME + "/config/"

# Load hydra test_config
with initialize(version_base=None, config_path=config_path):
    cfg = compose(config_name="test_config.yaml")
    hparams = cfg.experiment
n_classes = len(hparams["classes"])
n_samples = hparams["n_samples"]
test_size = hparams["test_size"]
val_size = hparams["val_size"]
unittest_data_folder = hparams["dataset_path"]
n_train_per_file = int((1 - test_size) * (1 - val_size) * n_samples)
n_val_per_file = int((1 - test_size) * (val_size) * n_samples)
n_test_per_file = int(test_size * n_samples)

# Load pt data files
if os.path.exists(processed_data_folder):
    train_dataset = []
    val_dataset = []
    test_dataset = []
    for i in range(n_classes):
        train_dataset.append(torch.load(processed_data_folder + f"train_data_{i}.pt"))
        val_dataset.append(torch.load(processed_data_folder + f"val_data_{i}.pt"))
        test_dataset.append(torch.load(processed_data_folder + f"test_data_{i}.pt"))
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)

# Load unit test dataset
if os.path.exists(unittest_data_folder):
    unittest_dataset = []
    for i in range(n_classes):
        unittest_dataset.append(torch.load(unittest_data_folder + f"/unittest_data_{i}.pt"))
    unittest_dataset = torch.utils.data.ConcatDataset(unittest_dataset)


# Skip test if data files are not found
# Assert length of dataset loaded
@pytest.mark.skipif(not os.path.exists(processed_data_folder), reason="Data files not found")
def test_data_length():
    expected_train = n_classes * n_train_per_file
    expected_val = n_classes * n_val_per_file
    expected_test = n_classes * n_test_per_file
    assert len(train_dataset) == expected_train, f"Trainset should have {expected_train} datapoints"
    assert len(val_dataset) == expected_val, f"Valset should have {expected_val} datapoints"
    assert len(test_dataset) == expected_test, f"Testset should have {expected_test} datapoints"


@pytest.mark.skipif(not os.path.exists(processed_data_folder), reason="Data files not found")
def test_data_shape():
    # # Assert each processed tensor has shape 1x224x224
    for i in range(len(train_dataset)):
        #     # images : 1,224,224
        #     # labels: 1
        #     # dataset: (images, labels) = ([1,1,224,224], 1)
        assert train_dataset[i][0].shape == (1, 224, 224), "Train image should be 1,224,224"

    for i in range(len(test_dataset)):
        assert test_dataset[i][0].shape == (1, 224, 224), "Test image should be 28x28"


@pytest.mark.skipif(not os.path.exists(processed_data_folder), reason="Data files not found")
def test_data_labels():
    # Assert that all labels are represented
    train_labels = [label.item() for _, label in train_dataset]
    train_labels = set(train_labels)
    test_labels = [label.item() for _, label in test_dataset]
    test_labels = set(test_labels)
    assert len(train_labels) == 5, "Train labels should have 5 classes"
    assert len(test_labels) == 5, "Test labels should have 5 classes"


@pytest.mark.skipif(not os.path.exists(unittest_data_folder), reason="Data files not found")
def test_unittest_data():
    # Assert that all labels are represented
    unittest_labels = [label.item() for _, label in unittest_dataset]
    unittest_labels = set(unittest_labels)
    assert len(unittest_labels) == 5, "Unittest labels should have 5 classes"
    for i in range(len(unittest_dataset)):
        assert unittest_dataset[i][0].shape == (1, 224, 224), "Unittest image should be 1,224,224"
