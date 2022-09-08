## Here we create the PyTorch model
import torch
import torch.nn as nn

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
	"""

	def __init__(self, drop_rate=0.1,img_size=224, patch_size=16 ):
		super().__init__()

        
        # Patch embeddings based on 
        # https://github.com/rwightman/pytorch-image-models/blob/fa8c84eede55b36861460cc8ee6ac201c068df4d/timm/models/layers/patch_embed.py#L15


	    img_size = ...
        patch_size = ...
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)



        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.num_patches = num_patches
        if verbose == True:
            print('frequncy stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))

        
        # the linear projection layer # what to use for???
        # new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        # self.proj = new_proj


        # positional embedding

        new_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, self.original_embedding_dim))
        self.pos_embed = new_pos_embed
        trunc_normal_(new_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)



    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):

    	x = x.unsqueeze(1)
    	x = x.transpose(2, 3)

    	B = x.shape[0] # batch


        x = self.proj(x).flatten(2).transpose(1, 2)  # Linear projection of 1D patch embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dist_token = ... ?

        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        return x


if __name__ == '__main__':
	input_tdim = 399
    ast_mdl = ASTModel(input_tdim=input_tdim,label_dim=50)
    
    # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    print(test_output.shape)











