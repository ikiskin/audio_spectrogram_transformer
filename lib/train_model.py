from models import AST
import config
import torch
import os
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import accuracy_score



# Load feature pickle here:
cv_fold = 1
x_val = None  # For code compatibility

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training on {device}')

if torch.cuda.device_count() > 1:
    print("Using data parallel")
    ast_model = nn.DataParallel(ast_model, device_ids=list(range(torch.cuda.device_count())))

ast_model = ast_model.to(device)




### Training loop

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ast_model.parameters(), lr=0.001, momentum=0.9)



all_train_loss = []
all_train_acc = []
all_val_loss = []
all_val_acc = []
best_val_loss = np.inf
best_val_acc = -np.inf

# best_train_loss = np.inf
best_train_acc = -np.inf

best_epoch = -1
checkpoint_name = None
overrun_counter = 0

for e in range(config.n_epochs):
    train_loss = 0.0
    ast_model.train()
    print(f'Training on {device}')

    all_y = []
    all_y_pred = []
    for batch_i, data in enumerate(train_loader, 0):

        ##Necessary in order to handle single and multi input feature spaces
        x, y = data

        x = x.to(device).detach()
        y = y.to(device).detach()
        optimizer.zero_grad()
        y_pred = ast_model(x)
        loss = criterion(y_pred, torch.max(y, 1)[1])

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        all_y.append(y.cpu().detach())
        all_y_pred.append(y_pred.cpu().detach())

        del x
        del y

    all_train_loss.append(train_loss/len(train_loader))

    all_y = torch.cat(all_y)
    all_y_pred = torch.cat(all_y_pred)
    train_acc = accuracy_score(np.argmax(all_y.detach().numpy(), axis=1),
              np.argmax(all_y_pred.detach().numpy(), axis=1))
    all_train_acc.append(train_acc)


    # Can add more conditions to support loss instead of accuracy. Use *-1 for loss inequality instead of acc
    if x_val is not None:
        val_loss, val_acc = test_model(model, val_loader, criterion, 0.5, device=device)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)

        acc_metric = val_acc
        best_acc_metric = best_val_acc
    else:
        acc_metric = train_acc
        best_acc_metric = best_train_acc
    if acc_metric > best_acc_metric:  
        # if checkpoint_name is not None:
            # os.path.join(os.path.pardir, 'models', 'pytorch', checkpoint_name)

        checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'

        torch.save(ast_model.state_dict(), os.path.join(config.model_dir, checkpoint_name))
        print('Saving model to:', os.path.join(config.model_dir, checkpoint_name)) 
        best_epoch = e
        best_train_acc = train_acc
        best_train_loss = train_loss
        if x_val is not None:
            best_val_acc = val_acc
            best_val_loss = val_loss
        overrun_counter = -1

    overrun_counter += 1
    if x_val is not None:
        print('Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, Val Acc: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_acc, val_loss/len(val_loader), val_acc,  overrun_counter))
    else:
        print('Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_acc, overrun_counter))
    if overrun_counter > config.max_overrun:
        break




print('Finished Training')








