from torch.utils.data import Dataset
import Data_preprocessing as dp

class ERDataset(Dataset):
    
    def __init__(self, fpath, nleft):
        super(ERDataset, self).__init__()
        data, labels = dp.preprocess(fpath, nleft)
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]