import torch
import os
import click

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dir = "models"


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # Import model
    model = torch.load(os.path.join(models_dir, model_checkpoint))

    # Import data
    test_dataset = torch.load("data/processed/test.pt")
    test_data = test_dataset[0]
    test_labels = test_dataset[1]
    # Convert to dataloader (to convert to batches)
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_labels),
        batch_size=64,
        shuffle=False,
    )

    test_predictions = []
    test_labels = []

    # Evaluation loop
    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            test_predictions.append(output.argmax(dim=1).cpu())
            test_labels.append(labels.cpu())

        test_predictions = torch.cat(test_predictions, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

    print("Accuracy: ", (test_predictions == test_labels).float().mean().item())


if __name__ == "__main__":
    evaluate()
