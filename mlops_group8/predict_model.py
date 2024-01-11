import torch
import wandb

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    # wandb.init(
    #     project='rice_classification',
    #     entity='mlops_group8',
    #     job_type='predict',   # or 'eval'
    #     dir='./wandb_output'
    #     )
    # wandb.log({'prediction': model(dataloader)})

    return torch.cat([model(batch) for batch in dataloader], 0)
