import torch
import torch.nn as nn
import config
import collections
from itertools import repeat

# Helper modules:

def _ntuple(n):
	def parse(x):
		if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
			return x
		return tuple(repeat(x, n))
	return parse

class Mlp(nn.Module):
	""" MLP as used in Vision Transformer, MLP-Mixer and related networks
	"""
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		bias = _ntuple(2)(bias)
		drop_probs = _ntuple(2)(drop)

		self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
		self.act = act_layer()
		self.drop1 = nn.Dropout(drop_probs[0])
		self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
		self.drop2 = nn.Dropout(drop_probs[1])

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop1(x)
		x = self.fc2(x)
		x = self.drop2(x)
		return x

class DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
	"""
	def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob
		self.scale_by_keep = scale_by_keep

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

	def extra_repr(self):
		return f'drop_prob={round(self.drop_prob,3):0.3f}'

class LayerScale(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=False):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(dim))

	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)


class Block(nn.Module):

	def __init__(
			self, dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
			drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
		self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.norm2 = norm_layer(dim)
		self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
		self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


	def forward(self, x):
		# x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
		# x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
		return x

class TransformerBlocks:

	def __init__(self):
		super().__init__()

		self.blocks = nn.Sequential(*[Block(
			dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=4., qkv_bias=False, init_values=None,
			drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
		for i in range(config.depth)])


	# def forward(self, x):
	# 	x = self.forward_features(x)
	# 	x = self.forward_head(x)
	# 	return x





