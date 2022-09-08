## Here we create the PyTorch model
import torch
import torch.nn as nn
import collections
from itertools import repeat
import config
import numpy as np
import transformer_encoder

class AST(nn.Module):

	""" Create an Audio Spectrogram model from building blocks. 
	From AST paper: https://arxiv.org/pdf/2104.01778.pdf
	
	First, the input audio waveform of t seconds is converted into a sequence of 128-dimensional log Mel filterbank (fbank)
	features computed with a 25ms Hamming window every 10ms. This results in a 128×100t spectrogram as input to the AST.
	We then split the spectrogram into a sequence of N 16×16 patches with an overlap of 6 in both time and frequency dimension, 
	where N = 12[(100t − 16)/10] is the number of patches and the effective input sequence length for the Transformer.
	We flatten each 16×16 patch to a 1D patch embedding of size 768 using a linear projection layer.

	Following the paper, we:
	  * Take our input spectrograms
	  * Perform patch split with overlap
	  * Project linearly and encode
	  * Each patch embedding added with learnable position embedding
	  * CLS classification token prepended to sequence
	  * Output of CLS token used for classification with linear layer
	  * Transformer encoder: multiple attention heads, variable depth
	  * Final output of model.
	"""

	def __init__(self, drop_rate=0.1, patch_size=16, embed_dim=768, input_tdim=1000, input_fdim=128, fstride=10, tstride=10,
	 n_classes=50):
		super(AST, self).__init__()

		print('In super init')
		# Patch embeddings based on 
		# https://github.com/rwightman/pytorch-image-models/blob/fa8c84eede55b36861460cc8ee6ac201c068df4d/timm/models/layers/patch_embed.py#L15

		# From PyTorch:
		def _ntuple(n):
			def parse(x):
				if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
					return x
				return tuple(repeat(x, n))
			return parse


		in_chans = 1
		# img_size = _ntuple(2)(img_size)
		patch_size = _ntuple(2)(patch_size)
		# num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
		# self.img_size = img_size
		self.patch_size = patch_size
		# self.num_patches = num_patches
		# print('self num patches here', self.num_patches)
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


		# automatcially get the intermediate shape
		f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
		num_patches = f_dim * t_dim
		self.num_patches = num_patches
		print('self num patches here', self.num_patches)

		# positional embedding
		self.embed_dim = embed_dim
		embed_len = self.num_patches
		
		self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
		print('self.pos_embed in init', np.shape(self.pos_embed))
		self.original_embedding_dim = self.pos_embed.shape[2]
		print('original embedding dim', self.original_embedding_dim)	
		


		if config.debug:
			print('frequncy stride={:d}, time stride={:d}'.format(fstride, tstride))
			print('number of patches={:d}'.format(num_patches))

		

		# new_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, self.original_embedding_dim))
		new_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.original_embedding_dim))
		self.pos_embed = new_pos_embed
		nn.init.trunc_normal_(new_pos_embed, std=.02)
		print('drop rate', drop_rate)
		self.pos_drop = nn.Dropout(p=drop_rate)



		# Transformer encoder blocks:

		self.transformer = transformer_encoder.TransformerBlocks()




	def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
		test_input = torch.randn(1, 1, input_fdim, input_tdim)
		test_proj = self.proj
		test_out = test_proj(test_input)
		f_dim = test_out.shape[2]
		t_dim = test_out.shape[3]
		return f_dim, t_dim

	def forward(self, x):

		x = x.unsqueeze(1)
		print('x unsqueezed', np.shape(x))
		x = x.transpose(2, 3)
		print('x after transpose', np.shape(x))
		B = x.shape[0] # batch

		x = self.proj(x).flatten(2).transpose(1, 2)  # Linear projection of 1D patch embedding
		print('x shape after linear proj', np.shape(x))
		self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
		print('Shape for token', np.shape(self.cls_token))
		cls_tokens = self.cls_token.expand(B, -1, -1)
		print('shape tokens', np.shape(cls_tokens))

		x = torch.cat((cls_tokens, x), dim=1)
		print('x after torch cat', np.shape(x))
		print('self.pos_embed dims', np.shape(self.pos_embed))
		x = x + self.pos_embed
		x = self.pos_drop(x)


		# Transformer encoder here
		x = self.transformer.blocks(x)


		# Final linear layer

		x = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, config.n_classes))(x)

		return x