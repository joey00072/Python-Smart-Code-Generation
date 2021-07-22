import torch
from torch._C import device
from main import model,generate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load('models/model_e10.pth',map_location=device))    


inp="start"

while inp!='q':
    inp = input("Enter start code: ")
    out = generate(model,inp)
    print(out)