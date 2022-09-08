import os
from features_util import extract_FBANK
import config
import pandas as pd
import librosa # Only used for loading + resample.
import numpy as np
import pickle

def compute_fbank_for_split(data_dir, label_file, cv_fold):
	''' Function to resample data. '''

	data = []
	label = []
	# Read dataframe from label_file in config:
	label_df = pd.read_csv(label_file) 

	# Sub-select CV fold to process:
	fold_df = label_df[label_df.fold == cv_fold]

	print(fold_df.head())

	# Load and re-sample audio file. Librosa returns floats in [-1.0, 1.0].
	for index, row in fold_df.iterrows():
			filename = os.path.join(data_dir, row['filename'])
			print(filename)
			audio_file, sr = librosa.load(filename, sr=config.sr)
			# Convert to int16 as required in downstream tasks
			audio_file = (audio_file * 32767).astype(np.int16) 
			lms = extract_FBANK(audio_file, config.len_fft, config.win_len, config.hop_len, config.n_mel, config.sr)
			data.append(lms)
			label.append(row['target'])

	feat_cv_fold = {"X_train": data, "y_train": label}
	
	pickle_name = ('lms_cv_fold_' + str(cv_fold) + '_len_fft_' + str(config.len_fft) + '_win_len_' + str(config.win_len)
	+ '_hop_len_' + str(config.hop_len) + '_n_mel_' + str(config.n_mel) + '.pkl')  

	with open(os.path.join(config.dir_out_feat, pickle_name), 'wb') as f:
		pickle.dump(feat_cv_fold, f, protocol=4)
		print('Saved features to:', os.path.join(config.dir_out_feat, pickle_name))
	
	return


for i in [1, 2, 3, 4, 5]:
	# IF PICKLE NOT EXIST DO, to add if time permits:
	compute_fbank_for_split(config.data_dir, config.label_file, i)