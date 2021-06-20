import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import Dataset as ds
import utils

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
        nn.init.constant_(self.gate_layer.bias, gate_bias)

    def forward(self, x):
        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))
        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)
        return torch.add(multiplyed_gate_and_normal, multiplyed_gate_and_input)

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
        # self.highway_token_matching = HighwayNet(embedding_len) # highway network in token matching level
        # self.linear_token_matching = nn.Linear(embedding_len, 1) # linear layer used in token matching level
        
        # layers in attribute matching level
        self.attribute_embeddings_left = nn.Embedding(nleft, embedding_len) # attribute embeddings for left entity in attribute matching level
        self.attribute_embeddings_right = nn.Embedding(nright, embedding_len) # attribute embeddings for right entity in attribute matching level
        
        # layers in entity matching level
        input_size = (nleft + nright) * embedding_len
        output_size = (nleft + nright) * 200
        pi = torch.tensor(0.01) # used for intializing last linear layer
        self.linear1_entity_matching = nn.Linear(input_size, output_size) # first linear layer in entity matching level
        self.linear2_entity_matching = nn.Linear(output_size, 2) # second linear layer in entity matching level
        nn.init.constant_(self.linear2_entity_matching.bias, -torch.log((1 - pi) / pi).item())
        
        # used as comparison result of attributes with empty value, learned during training
        empty_attr_res = torch.Tensor(1, self.embedding_len)
        nn.init.xavier_uniform_(empty_attr_res, gain=nn.init.calculate_gain('relu'))
        self.empty_attr_res = nn.Parameter(empty_attr_res, requires_grad=True)
    
    # abs diff between left entity tokens and right entity tokens
    def element_wise_compare(self, left, right):
        left = left.view(left.size(0), left.size(1), 1, -1)
        right = right.view(right.size(0), 1, right.size(1), -1)
        compare_matrix = torch.abs(left - right)
        return compare_matrix # shape(batch_size, n_tokens_left, n_tokens_right, 768)
    
    # get the the weight of left entity token wrt each token in right entity
    def get_token_weights(self, compare_matrix, tokens_mask):
        highway_out = self.highway_token_matching(compare_matrix) # shape(batch_size, n_tokens_left, n_tokens_right, 768)
        tokens_mask = tokens_mask.view(tokens_mask.size(0), 1, -1, 1)
        weight_vector = self.masked_softmax(self.linear_token_matching(highway_out), tokens_mask, dim=2)
        return weight_vector.squeeze(3) # shape(batch_size, n_tokens_left, n_tokens_right)
    
    # get the token in the right entity that is most similar to left entity token
    def get_max(self, weight_matrix, compare_matrix):
        selection_matrix = weight_matrix.unsqueeze(3)
        out = torch.mul(compare_matrix, selection_matrix).sum(2)
        return out # shape(batch_size, n_tokens_left, 768)
    
    # token matching layer
    def token_matching(self, left_embeddings, right_embeddings, tokens_mask):
        compare_matrix = self.element_wise_compare(left_embeddings, right_embeddings)
        weight_matrix = self.get_token_weights(compare_matrix, tokens_mask)
        res_matrix = self.get_max(weight_matrix, compare_matrix)
        return res_matrix
    
    # attribute matching layer
    def attribute_matching(self, token_embeddings, field_embeddings, tokens_mask, attrs_mask):
        start, res = 0, []
        for i, v in enumerate(token_embeddings.values()):
            if(v.size(1) == 0):
                res.append(torch.zeros(v.size(0), 1, 768))
                continue
            field = field_embeddings(torch.tensor(i)).view(-1,1) # shape(768, 1)
            mask = tokens_mask[:, start:start+v.size(1)].view(-1, v.size(1), 1)
            weights = self.masked_softmax(torch.matmul(v, field), mask, dim=1) # shape(batch_size, n, 1)
            # compare_matrix = compare_result[:, start:start+v.size(1)] # shape(batch_size, n, 768)
            summary = torch.sum(torch.mul(v, weights), 1).view(v.size(0), 1, -1) # shape(batch_size, 1, 768)
            start+=v.size(1)
            res.append(summary)
        res = torch.cat(res, dim=1) # shape(batch_size, n_attr, 768)
        res[~attrs_mask, :] = self.empty_attr_res.view(-1)
        return res
    
    # entity matching layer
    def entity_matching(self, left, right):
        concat = torch.cat((left, right), dim=1).view(left.shape[0], -1) # shape(batch_size, (nleft + nright) * 768)
        linear1_out = F.relu(self.linear1_entity_matching(concat)) # shape(batch_size, (nleft + nright) * 200)
        linear2_out = self.linear2_entity_matching(linear1_out) # shape(batch_size, 2)
        # return F.log_softmax(linear2_out, dim=1)
        return linear2_out
    
    # gets left entity embeddings and right entity embeddings
    def get_entity_tokens(self, x):
        left, right = [], []
        for key in ['left_fields', 'right_fields']:
            for v in x[key].values():
                if key == 'left_fields':
                    left.append(v)
                else:
                    right.append(v)
        return torch.cat(left, dim=1), torch.cat(right, dim=1)
    
    # gets token mask to detect padding tokens and attribute mask to detect empty attributes
    def get_mask(self, x):
        tokens_mask, attrs_mask = [], []
        for v in x.values():
            token_mask = (v != 0).sum(dim=2) > 0
            attr_mask = token_mask.sum(dim=1, keepdim=True) > 0
            tokens_mask.append(token_mask)
            attrs_mask.append(attr_mask)
        return torch.cat(tokens_mask, dim=1), torch.cat(attrs_mask, dim=1) # shape(batch_size, n_tokens), shape(batch_size, n_attrs)
        
    # compute softmax using a mask
    def masked_softmax(self, tensor, mask, dim=-1):
        mask[mask == 0] = -1e10
        mask[mask == 1] = 0
        tensor = tensor + mask
        return torch.softmax(tensor, dim=dim)
    
    def forward(self, x):
        # left_embeddings, right_embeddings = self.get_entity_tokens(x)
        left_tokens_mask, left_attrs_mask = self.get_mask(x['left_fields'])
        right_tokens_mask, right_attrs_mask = self.get_mask(x['right_fields'])
        # left_compare_matrix = self.token_matching(left_embeddings, right_embeddings, right_tokens_mask)
        # right_compare_matrix = self.token_matching(right_embeddings, left_embeddings, left_tokens_mask)
        left_attributes_rep = self.attribute_matching(x['left_fields'], self.attribute_embeddings_left, left_tokens_mask, left_attrs_mask)
        right_attributes_rep = self.attribute_matching(x['right_fields'], self.attribute_embeddings_right, right_tokens_mask, right_attrs_mask)
        output = self.entity_matching(left_attributes_rep, right_attributes_rep)
        return output
    
    # fits the model to the training data
    def run_train(self, 
                  train, 
                  validation,
                  best_val_path,
                  best_train_path=None,
                  num_epochs=40,
                  batch_size=32,
                  lr=0.01,
                  best_f1=0.0,
                  loss=None,
                  optimizer=None,
                  scheduler=None):
        
        dataset = ds.ERDataset(train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ds.ERDataset.collate_fn)
        
        if(optimizer == None):
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if(scheduler == None):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        if(loss == None):
            weight = self.get_class_weights(dataset.data['neg_pos'])
            loss = utils.FocalLoss(gamma=2, alpha=weight)
        
        train_plot_stats, val_plot_stats = [], []
        for epoch in range(num_epochs):
            Y, Y_hat = [], []
            for num, data in enumerate(loader):
                output = self(data)
                l = loss(output, data['labels'])
                l.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (num + 1) % 10 == 0:
                    print(f'epoch: {epoch+1} / {num_epochs}, step: {num+1} / {len(loader)}, loss: {l:.5f}')
                Y.extend(data['labels'].tolist())
                Y_hat.extend(output.detach().argmax(dim=1).view(-1).tolist())
            train_stats = self.get_stats(Y, Y_hat)
            self.log_stats(train_stats, "Training")
            if(best_train_path != None):
                self.save_model(epoch, train_stats, self.state_dict(), optimizer.state_dict(), best_train_path)
            val_stats= self.run_eval(validation, "Validation")
            if(val_stats[3] >= best_f1):
                best_f1 = val_stats[3]
                self.save_model(epoch, val_stats, self.state_dict(), optimizer.state_dict(), best_val_path)
            train_plot_stats.append(train_stats)
            val_plot_stats.append(val_stats)
            scheduler.step()
        utils.plot_stats(list(zip(*train_plot_stats)), list(zip(*val_plot_stats)))
            
    # evaluates the model on a dataset
    def run_eval(self, path, mode):
        dataset = ds.ERDataset(path)
        loader = DataLoader(dataset, shuffle=False, collate_fn=ds.ERDataset.collate_fn)
        print(f"evaluating model on {mode} set....")
        Y_hat, Y = [], [] 
        with torch.no_grad():
            for data in loader:
                output = self(data)
                Y_hat.append(torch.argmax(output).item())
                Y.append(data['labels'].item())
        stats = self.get_stats(Y, Y_hat)
        self.log_stats(stats, mode)
        return stats
    
    def get_stats(self, Y, Y_hat):
        accuracy = accuracy_score(Y, Y_hat) * 100
        recall = recall_score(Y, Y_hat) * 100
        precision = precision_score(Y, Y_hat) * 100
        f1 = f1_score(Y, Y_hat) * 100
        return accuracy, recall, precision, f1
    
    def get_class_weights(self, neg_pos):
        pos_neg_ratio = neg_pos[0] / neg_pos[1]
        pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
        neg_weight = 2 - pos_weight
        return [neg_weight, pos_weight]
    
    def save_model(self, epoch, stats, model_state, optim_state, path):
        checkpoint = {
            'epoch': epoch,
            'accuracy': stats[0],
            'recall': stats[1],
            'precision': stats[2],
            'f1': stats[3],
            'model_state': model_state,
            'optim_state': optim_state
            }
        torch.save(checkpoint, path)
        
    def log_stats(self, stats, type):
        print(f"{type}:\naccuracy: {stats[0]:.2f} recall: {stats[1]:.2f}, precision: {stats[2]:.2f}, f1: {stats[3]:.2f}")