#!/bin/bash
# Basic entrypoint

dvc pull -r remote_storage data.dvc
python -u mlops_group8/train_model.py
git rm -r --cached 'models'
dvc add models -f
dvc push -r remote_storage_models_train models.dvc

# TODO: dvc and git tracking needs to be updated.
# Either needs git authentication to push the models.dvc tracking file to git

# Execute the command passed into this entrypoint
#exec "$@
