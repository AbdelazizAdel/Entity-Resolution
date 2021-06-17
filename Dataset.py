import torch
from torch.utils.data import Dataset
import Data_preprocessing as dp

class ERDataset(Dataset):
    
    def __init__(self, fpath):
        super(ERDataset, self).__init__()
        data = dp.load_data(fpath)
        self.data = data
        
    def __len__(self):
        return len(self.data['labels'])
    
    def __getitem__(self, idx):
        example = {
            'left_fields': {k: v[idx] for k,v in self.data['left_fields'].items()},
            'right_fields': {k: v[idx] for k,v in self.data['right_fields'].items()},
            'labels': self.data['labels'][idx]
        }
        return example
    
    @staticmethod
    def collate_fn(batch):
        res = {
                'left_fields': {},
                'right_fields': {},
                'left_length': [],
                'right_length': [],
                'labels': []
              }
        
        for example in batch:
            left, right = [], []
            for key in ['left_fields', 'right_fields']:
                for k, v in example[key].items():
                    if k not in res[key]:
                        res[key][k] = []
                    res[key][k].append(torch.from_numpy(v))
                    if key == 'left_fields':
                        left.append(v.shape[0])
                    else:
                        right.append(v.shape[0])
            res['labels'].append(example['labels'])
            res['left_length'].append(left)
            res['right_length'].append(right)
            
        for key in ['left_fields', 'right_fields']:
            for k, v in res[key].items():
                res[key][k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
                
        res['labels'] = torch.tensor(res['labels'])
        res['left_length'] = torch.tensor(res['left_length'])
        res['right_length'] = torch.tensor(res['right_length'])
        return res