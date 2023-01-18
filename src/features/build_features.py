# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:55:04 2023

@author: Lenovo
"""

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd  # type: ignore
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer



@hydra.main(config_path="./../../config", config_name="default_config.yaml")
def main(cfg: DictConfig) -> None:
    """Converts .CSV files into tokenized PyTorch tensors"""
    logger = logging.getLogger(__name__)
    logger.info("Tokenize")
    c = cfg.build_features
    assert (
        c.split_train + c.split_test + c.split_eval == 1
    ), "The split train:{c.split_train} test:{c.split_test} is not possible"

    # %% Fetch Data
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "interim")

    train = pd.read_csv(
        os.path.join(data_path, "train.csv")
    )
    # test = pd.read_csv(
    #     os.path.join(data_path, "test.csv"), dtype={"id": np.int16, "target": np.int8}
    # )

    df_train = train
    # df_test = test

    premises = df_train.premise.values
    hypothesis = df_train.hypothesis.values
    labels = df_train.label.values
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(cfg.model["pretrained-model"])
    
    input_tok = []
    for i in range(len(premises)):
        input_tok.append((premises[i],hypothesis[i]))
        
    indices = tokenizer.batch_encode_plus(
        input_tok,
        max_length=c["max_sequence_length"],
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
        )
    
    input_ids = indices["input_ids"]
    attention_masks = indices["attention_mask"]

    train_inputs, rest_inputs, train_labels, rest_labels = train_test_split(
       input_ids, labels, random_state=42, test_size=c.split_test + c.split_eval
    )
    validation_inputs, eval_inputs, validation_labels, eval_labels = train_test_split(
       rest_inputs,
       rest_labels,
       random_state=42,
       test_size=c.split_test / (c.split_test + c.split_eval),
    )

    train_masks, rest_masks, _, rest_labels = train_test_split(
       attention_masks, labels, random_state=42, test_size=c.split_test + c.split_eval
    )

    validation_masks, eval_masks, _, _ = train_test_split(
       rest_masks,
       rest_labels,
       random_state=42,
       test_size=c.split_test / (c.split_test + c.split_eval),
    )

   # %% Convert to tensor
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    eval_inputs = torch.tensor(eval_inputs)
    eval_labels = torch.tensor(eval_labels, dtype=torch.long)
    train_masks = torch.tensor(train_masks, dtype=torch.long)
    validation_masks = torch.tensor(validation_masks, dtype=torch.long)
    eval_masks = torch.tensor(eval_masks, dtype=torch.long)

   # %% Save to file
    data_path = os.path.join(hydra.utils.get_original_cwd(), c.path, "processed")
    torch.save(train_inputs, os.path.join(data_path, "train_inputs.pkl"))
    torch.save(train_labels, os.path.join(data_path, "train_labels.pkl"))
    torch.save(validation_inputs, os.path.join(data_path, "validation_inputs.pkl"))
    torch.save(validation_labels, os.path.join(data_path, "validation_labels.pkl"))
    torch.save(train_masks, os.path.join(data_path, "train_masks.pkl"))
    torch.save(validation_masks, os.path.join(data_path, "validation_masks.pkl"))
    torch.save(eval_inputs, os.path.join(data_path, "eval_inputs.pkl"))
    torch.save(eval_labels, os.path.join(data_path, "eval_labels.pkl"))
    torch.save(eval_masks, os.path.join(data_path, "eval_masks.pkl"))
    logger.info("Finished! Output saved to '{}'".format(data_path))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()