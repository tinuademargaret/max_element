#%%
import os
import sys
from functools import partial
import json
from typing import List, Tuple, Union, Optional, Callable, Dict
import torch as t
from torch import Tensor
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import einops
from tqdm import tqdm
from jaxtyping import Float, Int, Bool
from pathlib import Path
import pandas as pd
import circuitsvis as cv
import webbrowser
from IPython.display import display
from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm
from eindex import eindex

from model import create_model
from training import train, TrainArgs
from dataset import MaxElementDataset, MinElementDataset
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
dataset = MinElementDataset(size=10, list_len=10, max_value=15, seed=43)

print("Sequence = ", dataset[0])
print("Str toks = ", dataset.str_toks)

# %%
args = TrainArgs(
    list_len=10,
    max_value=50,
    trainset_size=150_000,
    valset_size=10_000,
    epochs=50,
    batch_size=512,
    lr_start=1e-3,
    lr_end=1e-4,
    weight_decay=0.005,
    seed=42,
    d_model=96,
    d_head=48,
    n_layers=1,
    n_heads=2,
    d_mlp=None,
    normalization_type="LN",
    use_wandb=False,
    device=device,
)
model = train(args)
# %%
N = 2
dataset = MinElementDataset(size=N, list_len=10, max_value=50, seed=45)

logits, cache = model.run_with_cache(dataset.toks)
logits: Tensor = logits[:, -2, :]

targets = dataset.toks[:, -1: None]
predicted = logits.argmax(dim=-1)

print(f'input {dataset.toks}')
print(f'predicted:{predicted}')
print(f'target:{targets}')

logprobs = logits.log_softmax(-1) # [batch vocab_out]
probs = logprobs.softmax(-1)

# print(probs.shape) # [batch vocab_out]
# print(targets.shape)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, targets, "batch [batch tok]")
probs_correct = eindex(probs, targets, "batch [batch tok]")

# print(f'probs_correct: {probs_correct}')

avg_cross_entropy_loss = -logprobs_correct.mean().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")

cv.attention.from_cache(
    cache = cache,
    tokens = dataset.str_toks,
    batch_idx = list(range(1)),
    radioitems = True,
    return_mode = "view",
    batch_labels = [" ".join(s) for s in dataset.str_toks],
    mode = "small",
)
# %%
# Save the model
filename = "max_element_model.pt"
t.save(model.state_dict(), filename)

# Check we can load in the model
model_loaded = create_model(
    list_len=10,
    max_value=50,
    seed=0,
    d_model=96,
    d_head=48,
    n_layers=1,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None
)
model_loaded.load_state_dict(t.load(filename))

