import os
import csv
import torch 
from transformers import BertTokenizer, BertModel
import nlpaug.augmenter.word as naw
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
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
def get_data(fpath, nleft, da=False):
    with open(fpath) as csv_file:
        file = csv.reader(csv_file)
        data, labels, fields = [], [], []
        neg_pos = [0, 0]
        for idx, row in enumerate(file):
            if(idx == 0):
                fields = row[2:]
                continue
            label = int(row[1])
            entry = (row[2:(2+nleft)], row[(2+nleft):])
            # if(da and label == 1):
            #     labels.extend([1, 1, 1, 1])
            #     neg_pos[label]+=4
            #     aug_left = augment_entity(entry[0])
            #     aug_right = augment_entity(entry[1])
            #     aug_entities = list(zip(aug_left, aug_right))
            #     entry_2 = (entry[1], entry[0])
            #     data.extend([entry, entry_2, aug_entities[0], aug_entities[1]])
            # else:
            labels.append(label)
            neg_pos[label]+=1
            data.append(entry)
    return data, labels, fields, neg_pos

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
        left = [remove_stopwords(x) for x in left]
        right = [remove_stopwords(x) for x in  right]
        left_spans = {k:len(tokenizer.tokenize(v)) for k,v in zip(fields[:nleft], left)}
        right_spans = {k:len(tokenizer.tokenize(v)) for k,v in zip(fields[nleft:], right)}
        tokenized_seq = tokenizer(' '.join(left), ' '.join(right), max_length=256, truncation=True, return_tensors='pt')
        encoded_seq = get_bert_output(bert(**tokenized_seq, output_hidden_states=True).hidden_states).squeeze(0).numpy()
        start = 1
        for field, span in left_spans.items():
            left_fields[field].append(encoded_seq[start:start+span])
            start+=span
        start+=1
        for field, span in right_spans.items():
            right_fields[field].append(encoded_seq[start:start+span])
            start+=span
        if((i + 1) % 100 == 0):
            print(f"example {i + 1}/{len(data)}")
    return {'left_fields': left_fields, 'right_fields': right_fields}


# saves the output of the pretrained transformer to disk
def preprocess(data_dir, nleft):
    for dataset in ['train', 'valid', 'test']:
        path = data_dir + dataset
        data, labels, fields, neg_pos = get_data(path+".csv", nleft)
        encoded_data = encode_data(data, fields, nleft)
        encoded_data['labels'] = labels
        encoded_data['fields'] = fields
        encoded_data['neg_pos'] = neg_pos
        torch.save(encoded_data, path+".pt")

#loads pretrained embeddings from the provided path
def load_data(fpath):
    fpath = fpath.replace(".csv", ".pt")
    if(os.path.isfile(fpath)):
        data = torch.load(fpath)
        return data
    raise Exception(f"{fpath} not found")
    
def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return ' '.join(tokens_without_sw)

def augment_entity(entity):
    aug1 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
    aug2 = naw.SynonymAug(aug_src='wordnet')
    entity1 = [aug1.augment(x) for x in entity]
    entity2 = [aug2.augment(x) for x in entity]
    return entity1, entity2

def get_bert_output(x):
    return sum([a * 0.25 for a in x[-4:]])

preprocess('data/walmart_amazon/', 5)