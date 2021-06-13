import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import Dataset as ds
import time

bert = BertModel.from_pretrained('bert-base-uncased') # pretrained transformer used to obtain embeddings (BERT in this case)


class HighwayNet(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=F.relu):
        super(HighwayNet, self).__init__()
        self.activation_function = activation_function
        self.gate_activation = nn.Sigmoid()
        self.normal_layer = nn.Linear(input_size, input_size)
        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):
        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))
        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)
        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)


class HierMatcher(nn.Module):
    
    def __init__(self,
                 nleft,
                 nright,
                 embedding_len=768):
        super(HierMatcher, self).__init__()
        
        # general dataset properties
        self.embedding_len = embedding_len # size of the embedding of each token (768 in case of BERT)
        self.nleft = nleft # number of attributes in the left entity
        self.nright = nright # number of attributes in the right entity
        
        # layers in token matching level
        self.highway_token_matching = HighwayNet(embedding_len) # highway network in token matching level
        self.linear_token_matching = nn.Linear(embedding_len, 1) # linear layer used in token matching level
        
        # layers in attribute matching level
        self.attribute_embeddings_left = nn.Embedding(nleft, embedding_len) # attribute embeddings for left entity in attribute matching level
        self.attribute_embeddings_right = nn.Embedding(nright, embedding_len) # attribute embeddings for right entity in attribute matching level
        
        # layers in entity matching level
        input_size = (nleft + nright) * embedding_len
        self.highway_entity_matching = HighwayNet(input_size) # highway network in entity matching layer
        self.linear_entity_matching = nn.Linear(input_size, 2) # linear layer in entity matching level
        
        # used as comparison result of attributes with empty value, learned during training
        empty_attr_res = torch.Tensor(1, self.embedding_len)
        nn.init.xavier_uniform_(empty_attr_res, gain=nn.init.calculate_gain('relu'))
        self.empty_attr_res = nn.Parameter(empty_attr_res, requires_grad=True)
    
    # abs diff between left entity token and all tokens in right entity
    def element_wise_compare(self, vector, matrix):
        compare_matrix = torch.abs(vector - matrix) # shape(n, 768)
        return compare_matrix
    
    # get the the weight of left entity token wrt each token in right entity
    def get_token_weights(self, compare_matrix):
        highway_out = self.highway_token_matching(compare_matrix) # shape(n, 768)
        weight_vector = F.softmax(self.linear_token_matching(highway_out), dim=0) # shape(n, 1)
        return weight_vector
    
    # get the token in the right entity that is most similar to left entity token
    def get_max(self, weight_vector, compare_matrix):
        idx = torch.argmax(weight_vector)
        return compare_matrix[idx,:] # shape(1, 768)
    
    # token matching layer
    def token_matching(self, left_embeddings, right_embeddings):
        res= []
        for i in range(left_embeddings.size(0)):
            compare_matrix = self.element_wise_compare(left_embeddings[i,:], right_embeddings)
            weight_vector = self.get_token_weights(compare_matrix)
            res_vector = self.get_max(weight_vector, compare_matrix)
            res.append(res_vector.flatten())
        res = torch.stack(res) # shape(n, 768)
        return res
    
    # attribute matching layer
    def attribute_matching(self, token_embeddings, field_embeddings, compare_result, n_attr, n_tokens):
        start, res = 0, []
        for i in range(n_attr):
            if(n_tokens[i].item() == 0):
                res.append(self.empty_attr_res.flatten())
                continue
            tokens = token_embeddings[start:start+n_tokens[i]] # shape(n, 768)
            field = field_embeddings(torch.tensor(i)).reshape(-1,1) # shape(768, 1)
            weights = F.softmax(torch.matmul(tokens, field), dim=0) # shape(n, 1)
            compare_matrix = compare_result[start:start+n_tokens[i]] # shape(n, 768)
            sum = torch.sum(torch.mul(weights, compare_matrix), 0) # shape(768)
            start+=n_tokens[i].item()
            res.append(sum)
        res = torch.stack(res) # shape(n, 768)
        return res
    
    # entity matching layer
    def entity_matching(self, left, right):
        concat = torch.cat((left, right), dim=0).reshape(1, -1) # shape(1, 768 * n)
        highway_out = self.highway_entity_matching(concat) # shape(1, 768 * n)
        linear_out = self.linear_entity_matching(highway_out) # shape(1, 2)
        return F.softmax(linear_out, dim=1).flatten() # shape(2)
    
    def forward(self, x):
        left_len =  x['left_len'][0]
        right_len = x['right_len'][0]
        left_attrs_len = x['left_attrs_len'][0]
        right_attrs_len = x['right_attrs_len'][0]
        with torch.no_grad():
            left_embeddings = bert(**x['encoded_seq']).last_hidden_state[0][1:left_len+1]
            right_embeddings = bert(**x['encoded_seq']).last_hidden_state[0][left_len+2:right_len+left_len+2]
        left_compare_matrix = self.token_matching(left_embeddings, right_embeddings)
        right_compare_matrix = self.token_matching(right_embeddings, left_embeddings)
        left_entity_attributes_rep = self.attribute_matching(left_embeddings, self.attribute_embeddings_left, left_compare_matrix, self.nleft, left_attrs_len)
        right_entity_attributes_rep = self.attribute_matching(right_embeddings, self.attribute_embeddings_right, right_compare_matrix, self.nright, right_attrs_len)
        output = self.entity_matching(left_entity_attributes_rep, right_entity_attributes_rep)
        return output
    
    # fits the model to the training data
    def run_train(self, 
                  train, 
                  validation, 
                  num_epochs=40,
                  batch_size=16,
                  lr=0.001,
                  loss=nn.BCELoss(),
                  optimizer=None,
                  scheduler=None):
        
        if(optimizer == None):
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if(scheduler == None):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            
        train_dataset = ds.ERDataset(train, self.nleft)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        validation_dataset = ds.ERDataset(validation, self.nleft)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
        path = 'best_model.pt'
        
        num_steps = len(train_loader)
        cur_loss, best_f1 = 0.0, 0.0
        for epoch in range(num_epochs):
            for num, (data, labels) in enumerate(train_loader):
                start = time.time()
                output = self(data)
                end = time.time()
                print(f"time: {end - start}")
                l = loss(output[0].reshape(-1), labels.float())
                l.backward()
                cur_loss+=l.item()
                if((num + 1) % batch_size == 0 or num + 1 == num_steps):
                    optimizer.step()
                    optimizer.zero_grad()
                if (num + 1) % 100 == 0:
                    print(f'epoch: {epoch+1} / {num_epochs}, step: {num+1} / {num_steps}, loss: {(cur_loss / 100):.5f}')
                    cur_loss = 0
            recall, precision, f1, val_loss = self.__run_validation(validation_loader)
            if(f1 > best_f1):
                best_f1 = f1
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.state_dict(),
                    'optim_state': optimizer.state_dict()
                }
                torch.save(checkpoint, path)
            scheduler.step()
            
            
    # evaluates the model on validation set 
    def __run_validation(self,
                         validation,
                         loss=nn.BCELoss()):
        
        with torch.no_grad():
            Y_hat, Y = [], []
            l = 0
            for data, labels in validation:
                output = self(data)
                l += loss(output[0].reshape(-1), labels.float()).item()
                Y_hat.append(round(output[0].item()))
                Y.append(labels.item())
            recall = recall_score(Y, Y_hat)
            precision = precision_score(Y, Y_hat)
            f1 = f1_score(Y, Y_hat)
            
        print(f"validation:\nrecall: {recall}, precision: {precision}, f1: {f1}, loss: {l / len(validation)}")
        return recall, precision, f1, l / len(validation)
        