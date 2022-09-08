import os
import config
import librosa  # Import for reading wave files and resampling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
			# Convert to int16 as required in downstream tasks
			audio_file = (audio_file * 32767).astype(np.int16) 

			print(audio_file)


	# print(label_df) 
	return audio_file

def extract_FBANK(signal, fft_len, sr):
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

	Implementation: the implementation was based on a mixture of available open source
	libraries, including Python Speech Features, librosa, and a custom implementation for 
	VGGish. Some definitions were taken from other sources (e.g. mel scale), so 
	the implementation will differ slightly from other libraries.


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
	# fft_len = 512


	# Calculate frames with helper function
	frames = signal_to_frame(signal, win_len, hop_len)
	if config.debug:
		plt.title('Signal')
		plt.plot(signal)
		plt.show()

		plt.title('Frames')
		plt.plot(frames)
		plt.show()
	print(np.shape(frames), frames, signal)

	# Calculate window
	window = hann_window(win_len)
	windowed_frames = window * frames

	# STFT magniute given as:
	stft = np.abs(np.fft.rfft(windowed_frames, fft_len))

	# mel weights for chosen parameters:

	mel_weights = create_log_mel_matrix(sr, n_spec_bins=stft.shape[1]) # Can add arguments here
	

	# Take dot product, add small offset to prevent log errors
	log_mel_spectrogram = np.log(np.dot(stft, mel_weights) + 10**(-10)) 

	if config.debug:
		print('Shape of LMS', np.shape(log_mel_spectrogram))
		plt.imshow(log_mel_spectrogram, aspect='auto')
		plt.show()
	return log_mel_spectrogram



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

	w = 0.5 * (1. - np.cos(2*np.pi / win_len * np.arange(win_len)))
		
	return w

def stft_mag(signal, fft_len, win_len, hop_len):
	""" Calculate short-time Fourier transform """


def hz_to_mel(Hz):
	""" Convert frequency values in Hz to to the mel scale.
	Works elementwise to return a mel value for each item in array 
	passed in as Hz. We use the scale as suggested in:

	https://books.google.co.uk/books?id=mHFQAAAAMAAJ&q=2595&redir_esc=y

	"""
	return 2595 * np.log10(1. + Hz / 700.0)

def create_log_mel_matrix(sr, n_spec_bins, n_mel=32):
	""" Compute matrix to display 2D representation of signal in the 
	log-mel spectrogram domain. For the purpose of this task we skip
	a few error checks on validity of parameters and enforce hard 
	constraints to the edges of the mel frequency bands 
	Parameters
	----------
	n_spec_bins : Number of spec bins, which is the size of the FFT
	divided by 2 + 1.

	n_mel : Desired number of log-mel coefficients of output

	"""

	# Create bins of spectrogram from 0 to fs/2
	spec_hz = np.linspace(0.0, sr / 2., n_spec_bins) 

	# Convert scaling to mel
	spec_mel = hz_to_mel(spec_hz)

	# Upper edge at fs/2 Hz
	# Lower edge at 0 Hz

	# We require 2 extra bins in linsspace to correctly center
	# lower and higher edges:

	band_edges_mel = np.linspace(0, hz_to_mel(sr/2.), n_mel + 2)
	mel_weights = np.zeros((n_spec_bins, n_mel))

	for i in range(n_mel):
		lower, center, upper = band_edges_mel[i:i + 3]

		lower = (spec_mel - lower) / (center - lower)
		upper = (upper - spec_mel) / (upper - center)

		mel_weights[:, i] = np.maximum(0.0, np.minimum(lower, upper))

	# Remove DC bin:

	mel_weights[:, :] = 0.0

	return mel_weights




audio_file = resample(config.data_dir, config.label_file, 1)

len_fft = 2 ** int(np.ceil(np.log(200) / np.log(2.0))) #

lms = extract_FBANK(audio_file, len_fft, config.sr)
# plt.plot(fft)
# plt.show()

# After break: check if sig in floats or in ints
# see if frames are calculated correctly:
# Calculate frame size based on seconds supplied
# !!!!Seems to output 0s: check after lunch