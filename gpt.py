with open("data/input.txt", "r") as file:
    text = file.read()

"""get unique characters in the data."""
chars = list(set(text))
vocab_size = len(chars)
#print(f"Unique characters: {chars}")
#print(f"Vocabulary size: {vocab_size}")

"""character level tokenization. Assigning integer to each unique character."""
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}

"""encoder and decoder functions."""
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

#print(encode("HI THERE. hi there."))
#print(decode(encode("HI THERE. hi there.")))


"""creating torch dataset."""
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data[:1000])  # print first 1000 characters as integers

"""train test split"""
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

"""data loading and blocking and batching"""
block_size = 8  # context length for predictions
torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

"""Implementing a simple bigram model using PyTorch."""

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __int__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

m = BigramLanguageModel(vocab_size)
out, loss = m(xb, yb)
print(out)
print(loss)
