# Deep CNN Repository

This repository contains implementations of deep convolutional neural network models.

## ResNet

The `resnet.py` module provides a PyTorch implementation of several ResNet
variants, including ResNet-18, ResNet-34, ResNet-50, ResNet-101, and
ResNet-152.

You can instantiate a model as follows:

```python
from resnet import resnet18
model = resnet18(num_classes=10)
```

Running `resnet.py` as a script will create a ResNet-18 model and print its
architecture.
