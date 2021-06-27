import torch.optim as optim
import model as md
import torch
import os

train_path = 'data/amazon_google/train.csv'
valid_path = 'data/amazon_google/valid.csv'
test_path = 'data/amazon_google/test.csv'

best_val_path = 'data/amazon_google/best_model.pt'
best_train_path = 'data/amazon_google/train_best_model.pt'

best_f1 = 0.0
optimizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = md.HierMatcher(5, 5).to(device)

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
                lr=1e-3)