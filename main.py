import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import unidecode
import string
import tokenizers # [hugging face] pip install transformers
from model import RNN



chunk_len = 250
num_epochs = 500
batch_size = 1
print_every = 50
hidden_size = 256
num_layers = 2
lr = 0.003

# Getting all chars
tokenizer = tokenizers.ByteLevelBPETokenizer().from_file("tokenizer/vocab.json","tokenizer/merges.txt")
all_chars = string.printable
n_chars = tokenizer.get_vocab_size() 

# DataFile
with open("data/input.txt",encoding='utf-8') as f:
    file= f.read()

# Setting up GPU is avaliable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {'GPU (cuda)' if torch.cuda.is_available() else 'cpi'}")


def init_hidden(batch_size):
    hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    cell = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    return hidden, cell

def char_tensor( string):
    en = tokenizer.encode(string)
    return torch.tensor([x for x in en.ids])

def tensor_to_char(tensor_):
    return tokenizer.decode(tensor_.numpy())



def get_random_batch():
    start_idx = np.random.randint(0, len(file) - 3*chunk_len)
    end_idx = start_idx + 3*chunk_len + 1
    text_str = file[start_idx:end_idx]

    em_vec = char_tensor(text_str)
    if len(em_vec)<chunk_len+1:
        return get_random_batch()

    text_input = em_vec[0:chunk_len].view(1,chunk_len)
    text_target = em_vec[1:chunk_len+1].view(1,chunk_len)


    return text_input.long(), text_target.long()


def generate(model,initial_str="A", predict_len=100, temperature=0.85):
    hidden, cell = init_hidden(batch_size=batch_size)
    initial_input = char_tensor(initial_str)
    predicted = initial_str

    for p in range(len(initial_str) - 1):
        _, (hidden, cell) = model(
            initial_input[p].view(1).to(device), hidden, cell
        )

    last_char = initial_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = model(
            last_char.view(1).to(device), hidden, cell
        )
        output_dist = output.data.view(-1).div(temperature).exp()
        top_char = torch.multinomial(output_dist, 1)[0]
        curr_char = torch.tensor([int(top_char)])
        predicted_char = tensor_to_char(curr_char)
        predicted += predicted_char
        last_char = curr_char

    return predicted



def train(model):
    print("START...")

    for epoch in range(1, num_epochs + 1):
        inp, target = get_random_batch()
        hidden, cell = init_hidden(batch_size=batch_size)

        model.zero_grad()
        loss = 0
        inp = inp.to(device)
        target = target.to(device)

        for c in range(chunk_len):
            output, (hidden, cell) = model(inp[:, c], hidden, cell)
            loss += criterion(output, target[:, c])

        loss.backward()
        optimizer.step()
        loss = loss.item() / chunk_len

        if  epoch % print_every == 0:
            print(f"Loss: {loss}")
            print(generate(model))
        
        if epoch%(num_epochs//3)==0:
            ep_num = int(epoch/(num_epochs//10))
            print(f"Saving...{ep_num}")
            torch.save(model.state_dict(),f"models/model_e{ep_num}.pth")



model = RNN(n_chars, hidden_size, num_layers, n_chars).to(device)

if __name__=="__main__":

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train(model)