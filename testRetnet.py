import torch   
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

config = RetNetConfig(vocab_size = 64000)
retNet = RetNetDecoder(config)