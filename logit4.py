#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install torch


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Token and Model Initialization
token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token=token)

# Hook for capturing activations
activations = {}
def get_activation_hook(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks to capture MLP outputs of all layers
for i, layer in enumerate(model.model.layers):
    layer.mlp.register_forward_hook(get_activation_hook(f'layer_{i}_mlp'))

# Load GSM8K dataset
dataset = load_dataset("gsm8k", split="train[:10%]")  # Adjust split for testing

# Initialize storage for cross-entropy losses and KL divergences
cross_entropy_losses = []
kl_divergences = torch.zeros(32, 32)

# Define projection layer for logits comparison
hidden_size = model.config.hidden_size
vocab_size = model.config.vocab_size
linear_projection = nn.Linear(hidden_size, vocab_size).to(model.device)

# Loop through dataset examples
for example in dataset:
    # Tokenize input (taking 'question' from GSM8K)
    input_text = example['question']
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    # Forward pass through the model to get logits
    outputs = model(input_ids)
    final_logits = outputs.logits.squeeze(0)  # Remove batch dimension if necessary
    
    # Cross-entropy losses for each layer's MLP output
    for i in range(32):
        layer_logits = activations[f'layer_{i}_mlp']
        projected_logits = linear_projection(layer_logits)
        projected_logits = projected_logits.view(-1, vocab_size)
        final_logits_reshaped = final_logits.view(-1, vocab_size)
        
        # Compute Cross-Entropy loss
        loss = F.cross_entropy(projected_logits, final_logits_reshaped.argmax(dim=-1))
        cross_entropy_losses.append(loss.item())

    # KL divergence between layers
    for i in range(31):
        for j in range(i + 1, 32):
            logits_i = activations[f'layer_{i}_mlp']
            logits_j = activations[f'layer_{j}_mlp']
            
            projected_logits_i = linear_projection(logits_i).view(-1, vocab_size)
            projected_logits_j = linear_projection(logits_j).view(-1, vocab_size)
            
            kl_div = F.kl_div(F.log_softmax(projected_logits_i, dim=-1), 
                              F.softmax(projected_logits_j, dim=-1), 
                              reduction='batchmean')
            kl_divergences[i, j] += kl_div.item()

# Normalize KL divergences across the dataset
kl_divergences /= len(dataset)

# Plot KL divergence heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(kl_divergences.cpu().numpy(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("KL Divergence between LLaMA Layers on GSM8K Dataset")
plt.xlabel("Layer i")
plt.ylabel("Layer j")
plt.show()


# In[ ]:




