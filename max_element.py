#%%
import os
import sys
import torch as t
from pathlib import Path

from dataset import MaxElementDataset, MinElementDataset
from model import create_model
from plotly_utils import hist, bar, imshow, line, scatter

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

import einops
from eindex import eindex
from jaxtyping import Int, Float

from torch import Tensor

import functools
from tqdm import tqdm
from IPython.display import display
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv
# %%
dataset = MaxElementDataset(size=1, list_len=5, max_value=10, seed=42)

print(dataset[0].tolist())
print(dataset.str_toks[0])

# %%
filename = "max_element_model.pt"

model = create_model(
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

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);

#%%
min_filename = "min_element_model.pt"

min_model = create_model(
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

state_dict = t.load(min_filename)

state_dict = min_model.center_writing_weights(t.load(min_filename))
state_dict = min_model.center_unembed(state_dict)
state_dict = min_model.fold_layer_norm(state_dict)
state_dict = min_model.fold_value_biases(state_dict)
min_model.load_state_dict(state_dict, strict=False);
#%%
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

#%%
# %%
N = 500
dataset = MaxElementDataset(size=N, list_len=10, max_value=50, seed=43)

logits, cache = model.run_with_cache(dataset.toks)
logits: t.Tensor = logits[:, -2, :]

targets = dataset.toks[:, -1]
# %%
# Logit attribution
"""
Logit attribution

logits = W_U * ln(residiual)
residual = embed + ln(attn_out)
logits = (W_U * embed) + (W_U * attn_out) 
We would apply ln to the residual stack
"""
# Get W_U for answer tokens
answer_tokens = dataset.toks[:, -2]

logit_directions = model.tokens_to_residual_directions(answer_tokens)

assert(logit_directions.shape == t.Size([N, model.cfg.d_model]))
# %%
# function to project residual stream to logit direction 
def residual_stack_to_logit_dir(
        residual_stack: Float[Tensor, "... batch d_model"],
        cache: ActivationCache,
        logit_directions: Float[Tensor, "batch d_model"] = logit_directions

)-> Float[Tensor, "..."]:
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-2)

    average_logit = einops.einsum(
        scaled_residual_stack, logit_directions, 
        "... batch d_model, batch d_model -> ..."
    )/ N

    return average_logit
# %%
#%%
# Visualize accumulated logit 
accumulated_resid, labels = cache.accumulated_resid(layer=-1, pos_slice=-2, return_labels=True)
assert(accumulated_resid.shape == t.Size([2, N, model.cfg.d_model]))

logit_lens_dir = residual_stack_to_logit_dir(accumulated_resid, cache)

assert(logit_lens_dir.shape == t.Size([2]))


line(
    logit_lens_dir, 
    hovermode="x unified",
    title="Logits From Accumulated Residual Stream",
    labels={"x": "Layer", "y": "Logit"},
    xaxis_tickvals=labels,
    width=800
)
# %%
# Layer attribution
per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-2, return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_dir(per_layer_residual, cache)

line(
    per_layer_logit_diffs, 
    hovermode="x unified",
    title="Logits From Each Layer",
    labels={"x": "Layer", "y": "Logit"},
    xaxis_tickvals=labels,
    width=800
)
# %%
# Head attribution
per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-2, return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual, 
    "(layer head) ... -> layer head ...", 
    layer=model.cfg.n_layers
)
per_head_logit_diffs = residual_stack_to_logit_dir(per_head_residual, cache)

imshow(
    per_head_logit_diffs, 
    labels={"x":"Head", "y":"Layer"}, 
    title="Logits From Each Head",
    width=600
)
# %%
# Attention analysis
for layer in range(model.cfg.n_layers):
    attention_pattern = cache["pattern", layer][5]
    print(attention_pattern.shape)
    display(cv.attention.attention_patterns(tokens=dataset.str_toks[5], attention=attention_pattern))
# %%
print(dataset.str_toks[0])
# %%
# Zoom in on heads
attention_pattern = cache["pattern", layer][3]
print(attention_pattern.shape)
display(cv.attention.attention_heads(tokens=dataset.str_toks[3], attention=attention_pattern))
# %%
# Head ablation
import torch.nn.functional as F

def head_ablation_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_head"]:
    v[:, :, head_index_to_ablate, :] = 0.0

def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    # pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    loss = F.cross_entropy(
        log_probs,
        tokens
    )
    print(loss.shape)
    return loss
# -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    print(tokens.shape)
    loss_no_ablation = cross_entropy_loss(logits[:, -2, :], tokens[:, -1])

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("v", layer), temp_hook_fn)
            ])
            # Calculate the loss difference
            loss = cross_entropy_loss(ablated_logits[:, -2, :], tokens[:, -1])
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, dataset.toks[None, 0].to(device=model.cfg.device))

#%%
imshow(
    ablation_scores, 
    labels={"x": "Head", "y": "Layer", "color": "Loss diff"},
    title="Loss Difference After Ablating Heads", 
    text_auto=".2f",
    width=900, height=400
)
# %%
# OV & QK Circuits
# Expect QK to attend to the largest element seen

W_OV = model.W_V[0] @ model.W_O[0] # [head d_model_in d_model_out]

W_QK = model.W_Q[0] @ model.W_K[0].transpose(-1, -2) # [head d_model_dest d_model_src]

W_OV_full = model.W_E @ W_OV @ model.W_U

W_QK_full = model.W_E @ W_QK @ model.W_E.T

imshow(
    W_OV_full,
    labels = {"x": "Prediction", "y": "Source token"},
    title = "W<sub>OV</sub> for layer 1 (shows that the heads are copying)",
    width = 900,
    height = 500,
    facet_col = 0,
    facet_labels = [f"W<sub>OV</sub> [0.{h0}]" for h0 in range(model.cfg.n_heads)]
)

imshow(
    W_QK_full,
    labels = {"x": "Input token", "y": "Output logit"},
    title = "W<sub>QK</sub> for layer 1 (shows that the heads are attending to the largest element)",
    width = 900,
    height = 500,
    facet_col = 0,
    facet_labels = [f"W<sub>QK</sub> [0.{h0}]" for h0 in range(model.cfg.n_heads)]
)
# %%
# Attention pattern for min element
N = 500
dataset = MinElementDataset(size=N, list_len=10, max_value=50, seed=43)

logits, cache = model.run_with_cache(dataset.toks)
logits: t.Tensor = logits[:, -2, :]

targets = dataset.toks[:, -1]

for layer in range(min_model.cfg.n_layers):
    attention_pattern = cache["pattern", layer][5]
    print(attention_pattern.shape)
    display(cv.attention.attention_patterns(tokens=dataset.str_toks[5], attention=attention_pattern))