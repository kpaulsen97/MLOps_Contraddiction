# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:38:23 2023

@author: Lenovo
"""

# Note - you must have torchvision installed for this example
# -*- coding: utf-8 -*-
import logging
import os
import zipfile
from pathlib import Path

import hydra
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig



@hydra.main(config_path="./../../config", config_name="default_config.yaml")
def main(cfg: DictConfig) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading dataset from kaggle")

    dataset_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.path)
    zip_folder = os.path.join(dataset_path, "raw")

    try:
        import kaggle  # type: ignore
    except Exception:
        logger.warning(
            "Must athenticate the kaggle api according to https://www.kaggle.com/docs/api"  # noqa: E501
        )
        exit(1)

    try:
        kaggle.api.competition_download_files("contradictory-my-dear-watson", path=zip_folder)
    except Exception:
        logger.warning(
            "Must join the challange at: https://www.kaggle.com/c/contradictory-my-dear-watson/data"  # noqa: E501
        )
        exit(1)

    out_folder_raw = os.path.join(dataset_path, "interim")
    os.makedirs(out_folder_raw, exist_ok=True)
    with zipfile.ZipFile(
        os.path.join(zip_folder, "contradictory-my-dear-watson.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall(out_folder_raw)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()