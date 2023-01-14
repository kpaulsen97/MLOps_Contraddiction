# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:18:20 2023

@author: Lenovo
"""

import io

import torch
from google.cloud import storage
from transformers import XLMRobertaTokenizer

BUCKET_NAME = "cloud_function_model"
MODEL_FILE = "deployable_model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = blob.download_as_string()
my_model = io.BytesIO(my_model)

model = torch.jit.load(my_model)
model.eval()
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")



def encode(premise: str, hypothesis: str) -> list[int]:
    indices = tokenizer.encode_plus(
        premise, hypothesis,
        max_length=254,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
    )
    return indices["input_ids"], indices["attention_mask"]


def predict(request):
    request_json = request.get_json()
    premise = request_json["premise"]
    hypothesis = request_json["hypothesis"]
    input_model, mask = encode(premise, hypothesis)
    input_model = torch.IntTensor(input_model)
    mask = torch.LongTensor(mask)
    input_model.unsqueeze_(0)
    mask.unsqueeze_(0)
    (pred,) = model(input_model, mask)
    pred = torch.argmax(pred, 1).item()
    
    if pred == 0:
        answer = "The two sentences entails each other"

    elif pred == 1:
        answer = "The two sentences are unrelated"

    elif pred == 2:
        answer = "The two sentences contraddict each other"
    
    else:
        answer = "error"


    
    return answer


print("init done")