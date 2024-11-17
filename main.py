from pathlib import Path

import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

# checkpoint = "nreimers/TinyBERT_L-4_H-312_v2"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)

# input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt", truncation=True, padding=True)["input_ids"]
# outputs = model(input_ids)
# last_hidden_states = outputs.last_hidden_state

df = pd.read_csv("./data/text.csv")
