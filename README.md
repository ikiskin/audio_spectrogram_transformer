# Audio Spectrogram Transfer

## Training a custom class of AST (Audio Spectrogram Transformer) with log-mel filterbank features on ESC-50 data
Reference paper: [Audio Spectrogram Transformer](https://arxiv.org/pdf/2104.01778.pdf)

### Components:

* Custom implementation of FBANK features, loosely based on python speech features, librosa, and the vggish implementation
* Custom implementation of the core components of the AST, namely:
	* Patch creation
	* Linear projection
	* Positional token creation and embeddings
	* A minimal transformer/attention building block
	* Final linear layer and classification mapping



## How to run

### Requirements
Requirements here etc

### Structure
The code is configured in `lib/config.py`, which includes parameters that define the directories for saving outputs, and parameters that control feature transformation, the transformer architecture, and the training loop (e.g. batch size, learning rate, epochs).

### Log-mel features
The features are parameterised in lengths in units of samples, in `config.py`. By default, we use a Hanning window of 25 ms every 10 ms, and an FFT size of 512, with 128 log-mel coefficients. This creates an example feature representation as follows:

We also include per-sample normalisation (`norm_per_sample`), which removes the mean and divides each spectrogram input by its standard deviation.

### Transformer

Unlike in the original code accompanying the paper, the transformers implemented here include only the minimal core components, with no pre-training. The AST does not use the ViT building blocks, and hence is a simple minimal implementation. You can build on this implementation by adding any building blocks as desired to increase complexity.
The encoder properties are set with `embed_dim`, `num_heads`, and `depth` in `lib/config.py`.

## Model training and evaluation
The model is trained with a 5-fold validation strategy, where the model is trained on 80% of the training data, e.g. splits 1, 2, 3, 4, and tested on the remaining 20% (split 5). This procedure is iterated such that the model performance over `cv_fold` `i` is evaluated by training on all the remaining `cv_fold`s except `i`.

## Known issues/working notes:
* Paper uses Hamming window, Hanning implemented here
* The model achieves good performance on training data, but more time is needed to generalise well across splits.
* As a result of the way frames are calculated, we need to verify that the time dimensions of FBANK features match the ones in the original paper. This may affect good parameter choices for the embedding and token lengths.
