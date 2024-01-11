import hydra
import wandb
import random

@hydra.main(version_base=None, config_path='config', config_name='default_config.yaml')
def train(cfg):
    hparams = cfg.experiment
    # Read hyperparameters for experiment
    dataset_path = hparams['dataset_path']
    epochs = hparams['epochs']
    lr = hparams['lr']
    batch_size = hparams['batch_size']
    latent_dim = hparams['latent_dim']
    hidden_dim = hparams['hidden_dim']
    x_dim = hparams['x_dim']
    seed = hparams['seed']

    # Initialize wandb
    wandb_cfg = {
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'x_dim': x_dim,
        # 'seed': seed,
    }
    wandb.init(
        project='train_test',
        entity='mlops_group8',
        config=wandb_cfg,
        job_type='train',
        dir='./outputs',
    )

    # Simulate training
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # Log metrics to wandb
        wandb.log({'acc': acc, 'loss': loss})


if __name__ == '__main__':
    train()
