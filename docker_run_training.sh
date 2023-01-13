#!/bin/bash
# exit when any command fails
set -e
dvc pull
python3.9 -u src/features/build_features.py
python3.9 -u src/models/train_model.py
gsutil -m cp -r outputs gs://dtu_contraddiction_training
