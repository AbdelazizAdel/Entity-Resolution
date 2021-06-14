import os
import csv
import torch 
from transformers import BertTokenizer, BertModel

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)

'''
Args:
    fpath: string, the path to the dataset
    num_left: int, the number of attributes in the left entity
output:
    data: [([left, right)], list of tuple of lists where left is a list of the values of 
    the attributes of the left entity and right is a list of the values of the attributes
    of the right entity
    labels: list of int, the labels of the data, either 0 or 1
    fields: list of attribute names
'''
def get_data(fpath, nleft):
    with open(fpath) as csv_file:
        file = csv.reader(csv_file)
        data, labels, fields = [], [], []
        for idx, row in enumerate(file):
            if(idx == 0):
                fields = row[2:]
                continue
            labels.append(int(row[1]))
            data.append((row[2:(2+nleft)], row[(2+nleft):]))
    return data, labels, fields

'''
Args:
    data: [([], [])], output from get_data function
    fields: [str] attribute names
    nleft: int number of attributes in left entity
output:
    left_fields: {str:[[]]} dict containing each attribute of the left entity as key and
    the value is a list of lists where each list contains the embeddings of the tokens in
    the this attribute
    right_fields: same as left_fields but for the right entity
'''
@torch.no_grad()
def encode_data(data, fields, nleft):
    left_fields = {k:[] for k in fields[:nleft]}
    right_fields = {k:[] for k in fields[nleft:]}
    for i, (left, right) in enumerate(data):
        left_spans = {k:len(tokenizer.tokenize(v)) for k,v in zip(fields[:nleft], left)}
        right_spans = {k:len(tokenizer.tokenize(v)) for k,v in zip(fields[nleft:], right)}
        tokenized_seq = tokenizer(' '.join(left), ' '.join(right), max_length=256, truncation=True, return_tensors='pt')
        encoded_seq = bert(**tokenized_seq).last_hidden_state.squeeze().numpy()
        start = 1
        for field, span in left_spans.items():
            left_fields[field].append(encoded_seq[start:start+span])
            start+=span
        start+=1
        for field, span in right_spans.items():
            right_fields[field].append(encoded_seq[start:start+span])
            start+=span
        if(i % 100 == 0):
            print(f"example {i}/{len(data)}")
    return {'left_fields': left_fields, 'right_fields': right_fields}


# saves the output of the pretrained transformer to disk
def preprocess(data_dir, nleft):
    for dataset in ['train', 'valid', 'test']:
        path = data_dir + dataset
        data, labels, fields = get_data(path+".csv", nleft)
        encoded_data = encode_data(data, fields, nleft)
        encoded_data['labels'] = labels
        torch.save(encoded_data, path+".pt")

#loads pretrained embeddings from the provided path
def load_data(fpath):
    fpath = fpath.replace(".csv", ".pt")
    if(os.path.isfile(fpath)):
        data = torch.load(fpath)
        return data
    raise Exception(f"{fpath} not found")