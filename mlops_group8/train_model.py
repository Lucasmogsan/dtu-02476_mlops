import hydra
import wandb
import random

@hydra.main(config_path='mlops_group8/config', config_name='default_config.yaml')
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
        'epochs': cfg.epochs,
        'learning_rate': cfg.lr,
        'batch_size': cfg.batch_size,
        'latent_dim': cfg.latent_dim,
        'hidden_dim': cfg.hidden_dim,
        'x_dim': cfg.x_dim,
        # 'seed': cfg.seed,
    }
    wandb.init(
        project='rice_classification',
        entity='mlops_group8',
        config=wandb_cfg,
        job_type='train',   # or 'eval'
        dir='./wandb_output',
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
