import copy
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import fastchat 



def load_model(model_id=None):
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return model, tokenizer

