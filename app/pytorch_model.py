import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset

# set hyperparameters
input_size = 32827  
hidden_size = 64  
num_classes = 104 
num_epochs = 50  
learning_rate = 0.01  

# define model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.squeeze()
        return out

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        single_data = torch.from_numpy(self.data[index].todense()).float()
        return single_data

    def __len__(self):
        return self.data.shape[0]

def get_device():
    # select available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return device

def load_pytorch_model(model_path, device):
    # create model
    model = MLP(input_size, hidden_size, num_classes).to(device)
    print(next(model.parameters()).device)  
    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    # set to evaluation mode
    model.eval()
    return model