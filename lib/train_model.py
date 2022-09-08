from models import AST
import config
import torch
import os
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn


# Load feature pickle here:
cv_fold = 1

pickle_name = ('lms_cv_fold_' + str(cv_fold) + '_len_fft_' + str(config.len_fft) + '_win_len_' + str(config.win_len)
+ '_hop_len_' + str(config.hop_len) + '_n_mel_' + str(config.n_mel) + '.pkl')  

with open(os.path.join(config.dir_out_feat, pickle_name), 'rb') as f:
    feat = pickle.load(f)
    print('Loaded features from:', os.path.join(config.dir_out_feat, pickle_name))



# Convert to Torch format:
x_train = feat['X_train']
y_train = feat['y_train']
print(np.shape(x_train), np.shape(y_train))



# Dataloader
def build_dataloader(x_train, y_train, x_val=None, y_val=None, shuffle=True, n_channels=1):
    x_train = torch.tensor(x_train).float()
    if n_channels == 3:
        x_train = x_train.repeat(1,3,1,1)  # If using 3 channels e.g. vision transformers
    y_train = torch.tensor(y_train).to(torch.int64)
    y_train = torch.nn.functional.one_hot(y_train, num_classes=config.n_classes)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle)
    
    if x_val is not None:
        x_val = torch.tensor(x_val).float()
        if n_channels == 3:
            x_val = x_val.repeat(1,3,1,1)
        y_val = torch.tensor(y_val).to(torch.int64)
        y_val = torch.nn.functional.one_hot(y_val, num_classes=config.n_classes)
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle)

        return train_loader, val_loader
    return train_loader

train_loader = build_dataloader(x_train, y_train)

# Instantiate model

input_tdim = np.shape(x_train)[1] # n_frames
input_fdim = np.shape(x_train)[-1] # n_mel

ast_model = AST(input_tdim=input_tdim, n_classes=config.n_classes)

### Training loop

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ast_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print('inputs', inputs)
        # print('labels', labels)
        # print('actual input input shape', np.shape(inputs))
        # print('label shape', np.shape(labels))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = ast_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')


        checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'

        
        torch.save(ast_model.state_dict(), os.path.join(config.model_dir, 'pytorch', checkpoint_name))
        print('Saving model to:', os.path.join(config.model_dir, 'pytorch', checkpoint_name)) 

        running_loss = 0.0




print('Finished Training')








