import model as md
import torch
import os

train_path = 'E:/GUC/Semester 8/Bachelor/Papers Implementation/HierMatcher/data/walmart_amazon/train.csv'
valid_path = 'E:/GUC/Semester 8/Bachelor/Papers Implementation/HierMatcher/data/walmart_amazon/valid.csv'

model = md.HierMatcher(5, 5, embedding_len=768)

if(os.path.exists('./best_model.pt')):
    checkpoint = torch.load('./best_model.pt')
    model.load_state_dict(checkpoint['model_state'])

model.run_train(train_path, valid_path, num_epochs=1, batch_size=16, lr=0.001)