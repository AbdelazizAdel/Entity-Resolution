import csv
from transformers import BertTokenizer
import numpy as np

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

'''
Args:
    fpath: string, the path to the dataset
    num_left: int, the number of attributes in the left entity
output:
    data: [([left, right)], list of tuple of lists where left is a list of the values of the attributes of the left entity and right is a list
    of the values of the attributes of the right entity
    labels: list of int, the labels of the data, either 0 or 1
'''
def get_data(fpath, num_left):
    with open(fpath) as csv_file:
        file = csv.reader(csv_file)
        data, labels = [], []
        for idx, row in enumerate(file):
            if(idx == 0):
                continue
            labels.append(int(row[1]))
            data.append((row[2:(2+num_left)], row[(2+num_left):]))
    return data, labels

'''
Args:
    data: [([], [])], output from get_data function
    tokenizer: BertTokenizer, the tokenizer used to encode the data
output:
    res: [{}], list of encoded data, where each example is a dictionary conatining:
        1) encoded_seq: {}, the encoded data in the format expected by BERT
        2) left_attrs_len: [int], the number of tokens in each attribute of the left entity
        3) right_attrs_len: [int], the number of tokens in each attribute of the right entity
        4) left_len: int, the total number of tokens in the left entity without special characters
        5) right_len: int, the total number of tokens in the right entity without special characters
'''
def encode_data(data, tokenizer):
    res = []
    for left, right in data:
        left_tokens = [tokenizer.tokenize(s) for s in left]
        right_tokens = [tokenizer.tokenize(s) for s in right]
        left_attrs_len = np.array([len(attr) for attr in left_tokens], dtype=np.int64)
        right_attrs_len = np.array([len(attr) for attr in right_tokens], dtype=np.int64)
        encoded_seq = tokenizer(' '.join(left), ' '.join(right), max_length=256, padding='max_length')
        encoded_seq['input_ids'] = np.array(encoded_seq['input_ids'], dtype=np.int64)
        encoded_seq['token_type_ids'] = np.array(encoded_seq['token_type_ids'], dtype=np.int64)
        encoded_seq['attention_mask'] = np.array(encoded_seq['attention_mask'], dtype=np.int64)
        entry = {
            'encoded_seq': encoded_seq,
            'left_attrs_len': left_attrs_len,
            'right_attrs_len': right_attrs_len,
            'left_len': sum(left_attrs_len),
            'right_len': sum(right_attrs_len)
        }
        res.append(entry)
    return res

'''
Args:
    fpath: string, the path to the dataset
    num_left: int, the number of attributes in the left 
output:
    encoded_data: [{}], same as encode_data function
    labels: list of int, the labels of the data, either 0 or 1
'''
def preprocess(fpath, num_left):
    data, labels = get_data(fpath, num_left)
    encoded_data = encode_data(data, tokenizer)
    return encoded_data, labels


