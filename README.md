# fast-neural-style

July 2018 Update:

1. Upgrade to PyTorch 0.4.0.
2. Minor Refactor.
3. Allow assigning an independent style weight for each layer.
4. Provide a Dockerfile

The code to the [first blog post about this project ](https://medium.com/@ceshine/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902) can be found in tag **[201707](https://github.com/ceshine/fast-neural-style/tree/201707)**.

## Stylize Script Usage Example
```
python stylize.py models/model_rain_princess_cropped.pth content_images/pic.jpg pic-512.jpg --resize=512
```

## Old README

This personal fun project is heavily based on [abhiskk/fast-neural-style](https://github.com/abhiskk/fast-neural-style) with some changes and a video generation notebook/script:

1. Use the official pre-trained VGG model
2. Output intermediate results during training
3. Add Total Variation Regularization as described in the paper

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf).
