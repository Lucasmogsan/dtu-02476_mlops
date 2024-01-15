#!/bin/bash
# Basic entrypoint

dvc pull -r remote_storage data.dvc
python -u mlops_group8/train_model.py
dvc push -r remote_storage_models_train models.dvc


# Execute the command passed into this entrypoint
#exec "$@