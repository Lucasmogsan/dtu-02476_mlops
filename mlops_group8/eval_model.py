import torch
import os
from utility.util_functions import set_directories, load_data
import hydra
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_path, _, models_dir, _ = set_directories()


@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def evaluate(cfg):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")

    hparams = cfg.experiment
    model_name = hparams["model_name"]
    classes_to_eval = hparams["classes"]
    batch_size = hparams["batch_size"]

    print("Evaluating ", model_name)

    # Import model
    model = torch.load(os.path.join(models_dir, model_name + ".pt"))

    # Import data
    test_dataloader = load_data(classes_to_eval, batch_size, processed_path, train=False)

    test_predictions = []
    test_labels = []

    # Evaluation loop
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=False,
            on_trace_ready=tensorboard_trace_handler("./log/eva"),
        ) as prof:
            for i, batch in enumerate(test_dataloader):
                print(f"Iteration: {i+1}/{len(test_dataloader)}")
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)

                test_predictions.append(output.argmax(dim=1).cpu())
                test_labels.append(labels.cpu())

                prof.step()

        test_predictions = torch.cat(test_predictions, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

    print("Accuracy: ", (test_predictions == test_labels).float().mean().item())

    # Print from the 'prof' object created above:
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))


if __name__ == "__main__":
    evaluate()
