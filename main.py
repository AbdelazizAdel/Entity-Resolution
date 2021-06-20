import torch.optim as optim
import model as md
import torch
import os

train_path = 'data/walmart_amazon/train.csv'
valid_path = 'data/walmart_amazon/valid.csv'
test_path = 'data/walmart_amazon/test.csv'

best_val_path = 'data/walmart_amazon/best_model.pt'
best_train_path = 'data/walmart_amazon/train_best_model.pt'

best_f1 = 0.0
optimizer = None
model = md.HierMatcher(5, 5, embedding_len=768)

if(os.path.exists(best_val_path)):
    checkpoint = torch.load(best_val_path)
    best_f1 = checkpoint['f1']
    model.load_state_dict(checkpoint['model_state'])
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optim_state'])
        
model.run_train(train_path,
                valid_path,
                best_val_path,
                best_train_path=best_train_path,
                best_f1=best_f1,
                optimizer=optimizer,
                num_epochs=15,
                batch_size=32,
                lr=0.01)


#TODO: use oversampling to increase positive class using random replication or mixDA from Ditto