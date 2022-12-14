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

plot_dir = os.path.join(
	ROOT_DIR, "plots")
# Parameters for FBANK transform:

# From paper: "with a 25ms Hamming window every 10ms"
win_len = 200 # In samples. -> 25 ms
hop_len = 80 # In samples -> 10ms
len_fft = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0))) # Could move out of config
len_fft = 2*len_fft # To fill spectrum properly
sr = 8000 # Signal rate (Hz) for re-sampling:
n_mel = 128 # number of bins in log-mel spectrogram to match AST paper
norm_per_sample = True


# Choose model:
# model_name = 'conv' # Selects simple 2D conv (simple_cnn_baseline.py)
model_name = 'AST' # Any other string selects transformer: TODO, check string logic

# Transformer encoder properties
embed_dim = 768
num_heads = 8
depth = 4
n_classes = 50
dropout = 0.1
# Training loop properties
batch_size = 4  # Size 48 used in paper, with my GPU and this implementation, 4 was the sweet spot
n_epochs = 50
max_overrun = 20  # for early stopping, nb of epochs to train with no improvement in best metric
lr = 0.00001  # Learning rate used with optimiser (Adam)

# Evaluation
debug = False # debug parameter. Only set to True if issues suspected, opens various diagnostic
# plots and print statements

# Create directories if they do not exist:
for directory in [dir_out_feat, model_dir, plot_dir]:
	if not os.path.isdir(directory):
		os.mkdir(directory)
		print('Created directory:', directory)
