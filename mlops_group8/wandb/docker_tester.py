'''
At https://wandb.ai/home, click on the profile icon in the upper right corner
and then go to settings. Scroll down to the danger zone and generate a new API key.
Copy the API key and run the following command from the root directory.

docker build -f dockerfiles/wandb.dockerfile . -t wandb:latest
docker run -e WANDB_API_KEY=<your-api-key> wandb:latest
'''
import random
import wandb

wandb.init(
    project='wandb_docker_test',
    entity='mlops_group8',
    name='from-docker',
    dir='./outputs',
)
for _ in range(100):
    wandb.log({'test_metric': random.random()})
