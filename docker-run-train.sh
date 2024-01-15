#!/bin/bash
docker run -it \
    --name train \
    -v $(pwd)/models:/models \
    -v $(pwd)/data/processed:/data/processed \
    -v $(pwd)/reports/figures:/reports/figures \
    -e WANDB_API_KEY=<your-wandb-api-key> \
    mlops_trainer:latest
