import tokenizers
import torch
tokenizer = tokenizers.ByteLevelBPETokenizer()

paths=['data/input.txt']

tokenizer.train(
    
    files=paths,
    vocab_size=150,
    min_frequency=5,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)

inp="import numpy as np"

tokenizer.save_model("tokenizer")

t = tokenizer.encode(inp)

v = torch.tensor([item for item in t.ids])

print(v)
print([item for item in t.tokens])

print(tokenizer.decode(v.numpy()))

