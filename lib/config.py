import os
import numpy as np

# Configuration file for parameters
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
data_dir = os.path.join(
    ROOT_DIR, "data", "ESC-50-master", "audio")

label_file = os.path.join(
    ROOT_DIR, "data", "ESC-50-master", "meta", "esc50.csv")

dir_out_feat = os.path.join(
    ROOT_DIR, "data", "feat")

model_dir = os.path.join(
    ROOT_DIR, "models")
# Parameters for FBANK transform:

# From paper: "with a 25ms Hamming window every 10ms"
win_len = 200 # In samples. -> 25 ms
hop_len = 80 # In samples -> 10ms
len_fft = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0))) # Could move out of config
len_fft = 2*len_fft # To fill spectrum properly
sr = 8000 # Signal rate (Hz) for re-sampling:
n_mel = 128 # number of bins in log-mel spectrogram to match AST paper
norm_per_sample = True


# Transformer encoder properties
embed_dim = 768
num_heads = 4
depth = 2
n_classes = 50

# Training loop properties
batch_size = 10
n_epochs = 100
max_overrun = 50 # for early stopping, nb of epochs to train with no improvement



debug = False # debug parameter. Only set to True if issues suspected, opens various diagnostic
# plots and print statements

# Create directories if they do not exist:
for directory in [dir_out_feat, model_dir]:
	if not os.path.isdir(directory):
		os.mkdir(directory)
		print('Created directory:', directory)
