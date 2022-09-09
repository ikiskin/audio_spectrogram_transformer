
from models import AST
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import config
import os
import pickle
import torch

# Accuracy and evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_recall_curve, plot_precision_recall_curve



def evaluate_model(model, X_test, y_test, filename):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on {device}')

    x_test = torch.tensor(X_test).float()

    y_test = torch.tensor(y_test).to(torch.int64)
    y_test = torch.nn.functional.one_hot(y_test, num_classes=config.n_classes)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    y_preds_all = np.zeros([len(y_test), config.n_classes])
    model.eval() # Important to not leak info from batch norm layers and cause other issues


    all_y_pred = []
    all_y = []

    for x, y in test_loader:
        print('Ensuring correct dim', np.shape(x), np.shape(y))
        y_pred = model(x)
        # print(y_pred)
        print('in for loop', np.shape(y_pred))
        all_y.append(y)
        all_y_pred.append(y_pred)

        del x
        del y
        del y_pred

    all_y_pred = torch.cat(all_y_pred).detach().numpy()
    all_y = torch.cat(all_y).detach().numpy()

    print('shape of pred', np.shape(all_y), np.shape(all_y_pred))
    
    # print(s

    test_acc = accuracy_score(np.argmax(all_y, axis=1), np.argmax(all_y_pred, axis=1))
    print('Test accuracy', test_acc)
    print('Random guess', 1/50.)


    with open(os.path.join(config.plot_dir, filename + '_cm.txt' ), "w") as text_file:
        print(classification_report(np.argmax(all_y, axis=1), np.argmax(all_y_pred, axis=1)), file=text_file)
    return test_acc





def load_model(filepath, model):
    checkpoint = model.load_state_dict(torch.load(filepath), strict=False)

    return model


# Load best models: TODO: automate
checkpoint_names = {1:'model_e99_2022_09_09_08_56_09.pth', 2:'model_e96_2022_09_09_08_57_52.pth',
                     3:'model_e99_2022_09_09_08_59_44.pth', 4:'model_e95_2022_09_09_09_01_32.pth', 5:'model_e95_2022_09_09_09_03_24.pth'}

# Load feature pickle here:
for i, cv_fold in enumerate([1, 2, 3, 4, 5]):
    # Each model [i] has been trained on indices NOT i, so we just need to load by index [i]

    pickle_name = ('lms_cv_fold_' + str(cv_fold) + '_len_fft_' + str(config.len_fft) + '_win_len_' + str(config.win_len)
    + '_hop_len_' + str(config.hop_len) + '_n_mel_' + str(config.n_mel) + '.pkl')  

    with open(os.path.join(config.dir_out_feat, pickle_name), 'rb') as f:
        feat = pickle.load(f)
        print('Loaded features from:', os.path.join(config.dir_out_feat, pickle_name))

        x_test = feat['X_train']
        y_test = feat['y_train']

    checkpoint_name = checkpoint_names[cv_fold]
    print(checkpoint_name)
    # Convert to Torch format:

    print('X_test, y_test', np.shape(x_test), np.shape(y_test))

    input_tdim = np.shape(x_test)[1] # n_frames
    input_fdim = np.shape(x_test)[-1] # n_mel

    filepath = os.path.join(config.model_dir, str(cv_fold), checkpoint_name)
    evaluate_model(load_model(filepath, AST(input_tdim=input_tdim, n_classes=config.n_classes)), x_test, y_test,
     str(cv_fold) + '_' + checkpoint_name)