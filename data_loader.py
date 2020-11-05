import torch
from torch.utils.data import Dataset
class DocDataset(Dataset):
    def __init__(self, mode, list_of_dict):
        assert mode in ["train", "test"]
        self.mode = mode
        self.list_of_dict = list_of_dict
            
    def __getitem__(self, idx):
        inputid = self.list_of_dict[idx]['src']
        tokentype = self.list_of_dict[idx]['segs']
        attentionmask = self.list_of_dict[idx]['att_msk']
        inputid = torch.tensor(inputid)
        tokentype = torch.tensor(tokentype)
        attentionmask = torch.tensor(attentionmask)
        return inputid, tokentype, attentionmask
    
    def __len__(self):
        return len(self.list_of_dict)

class VecDataset(Dataset):
    def __init__(self, mode, list_of_dict):
        assert mode in ["train", "test"]
        self.mode = mode
        self.list_of_dict = list_of_dict
    def __getitem__(self, index):
        hidden_state = self.list_of_dict[index]["hidden_state"]
        hidden_state = torch.tensor(hidden_state)
        if (self.mode == "train"):
            label = self.list_of_dict[index]["label"]
            label = torch.tensor(label)
            return hidden_state, label
        else:
            return hidden_state
    def __len__(self):
        return len(self.list_of_dict)