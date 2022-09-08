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

# Parameters for FBANK transform:
win_len = 200 # In samples.
hop_len = 100 # In samples
len_fft = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0))) # Could move out of config
sr = 8000 # Signal rate (Hz) for re-sampling:
n_mel = 32 # number of bins in log-mel spectrogram

debug = False# debug parameter. Only set to True if issues suspected, opens various diagnostic
# plots and print statements

# Create directories if they do not exist:
for directory in [dir_out_feat]:
	if not os.path.isdir(directory):
		os.mkdir(directory)
		print('Created directory:', directory)