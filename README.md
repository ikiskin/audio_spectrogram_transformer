# Audio Spectrogram Transfer

## Training a custom class of AST (Audio Spectrogram Transformer) with log-mel filterbank features on ESC-50 data

Components:

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


### Transformer

Unlike in the original code accompanying the paper, the transformers implemented here include only the minimal core components, with no pre-training. The AST does not use the ViT building blocks, and hence is a simple minimal implementation. You can build on this implementation by adding any building blocks as desired to increase complexity.
The encoder properties are set with `embed_dim`, `num_heads`, and `depth` in `lib/config.py`.

## Model training and evaluation
The model is trained with a 5-fold validation strategy, where the model is trained on 80% of the training data, e.g. splits 1, 2, 3, 4, and tested on the remaining 20% (split 5). This procedure is iterated such that the model performance over `cv_fold` i is evaluated by training on all the remaining `cv_fold`s except i.

## Known issues/working notes:
* Paper uses Hamming window, Hanning implemented here
* The model achieves good performance on training data, but more time is needed to generalise well across splits.
* Dimensions of fbank need matching to paper