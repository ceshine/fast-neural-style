# fast-neural-style

This personal fun project is heavily based on [abhiskk/fast-neural-style](https://github.com/abhiskk/fast-neural-style) with some changes and a video generation notebook/script:

1. Use the official pre-trained VGG model
2. Output intermediate results during training
3. Add Total Variation Regularization as described in the paper

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/gtwnyp9n49lqs7t/saved-models.zip?dl=0).

[The blog post on this project](https://medium.com/@ceshine/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902)
