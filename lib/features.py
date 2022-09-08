import os
import config
import librosa  # Import for reading wave files and resampling
import pandas as pd
import numpy as np

def get_wav_data(data_dir):
	print(data_dir)
	for file in os.listdir(data_dir):
		if file.endswith('.wav'):
			print(file)




def resample(data_dir, label_file, cv_fold):
	''' Function to resample data. '''

	# Read dataframe from label_file in config:
	label_df = pd.read_csv(label_file) 

	# Sub-select CV fold to process:
	fold_df = label_df[label_df.fold == 1]

	# Load and re-sample audio file. Librosa returns floats in [-1.0, 1.0].
	for index, row in fold_df.iterrows():
		if index == 0:  # temp to help debugging
			filename = os.path.join(data_dir, row['filename'])
			print(filename)
			audio_file, sr = librosa.load(filename, sr=config.sr)
			print(audio_file)


	# print(label_df) 
	return audio_file

def extract_FBANK(signal):
	""" Extract FBANK features for wave audio file given in 8kHz. 
	
	We assume FBANK refers to log-mel filterbank features as referred to in literature.
	To compute these features, we need to compute an STFT (short-time Fourier transform),
	applying appropriate windowing functions to smooth edge discontinuities which result 
	in artificats due to the assumptions of periodicity in the FFT.

	Following the windowed STFT, we need to map the result to the logarithmic mel scale.
	The transform can be parameterised with options:
	* fft_len, length of FFT
	* win_len, the window length [in samples] for which we calculate the FFT.
	* hop_len, the length of the hop/stride [in samples] taken of the window

	Parameters
	----------
	signal : [-1, 1] float
		signal converted to 8 kHz mono.

	Returns
	-------

	M : np.ndarray [shape=...]
		Matrix of log-mel filterbank coefficients
	"""

	# Test run:
	win_len = 200
	hop_len = 100

	frames = signal_to_frame(signal, win_len, hop_len)

	print(np.shape(frames), frames)


def signal_to_frame(signal, win_len, hop_len):
	""" Calculate frames from signal. We need to convert our signal
	to frames to apply a windowing function to. Following this, we can compute
	the FFT of frames to build up the STFT.

	The number of frames, N, for a signal of length L will be:

	N = 1 + (L - win_len) / hop_len.

	We perform rounding to return integers.

	Please note that this function uses stride_tricks.as_strided and can be replaced 
	by numpy.lib.stride_tricks.sliding_window_view for safer implementation in future.

	The original source code was modified for 1D arrays by removing some excess code.


	"""
	L = len(signal)
	N = 1 + int(np.floor((L - win_len) / hop_len))
	shape = (N, win_len)
	strides = (signal.strides[0] * hop_len,) + signal.strides

	return np.lib.stride_tricks.as_strided(signal, shape = shape, strides = strides)





def hann_window(win_len):
	""" Implement periodic window for performing smoothing:	
	By definition:

	w = 0.5 * (1 - np.cos(2*np.pi*n/N))  0 <= n <= N 

	where N is the window length - 1, n is index of frame.

	We use the periodic version which is equivalent below. 
	(refer to https://www.mathworks.com/help/signal/ref/hann.html for formulae)

	"""

	w = 0.5 * (1. - np.cos(2*np.pi*np.arange(win_len)/win_len))
		
	return w

def stft_mag(signal, fft_len, win_len, hop_len):
	""" Calculate short-time Fourier transform """







audio_file = resample(config.data_dir, config.label_file, 1)
extract_FBANK(audio_file)

# After break: check if sig in floats or in ints
# see if frames are calculated correctly:
# Calculate frame size based on seconds supplied
# !!!!Seems to output 0s: check after lunch