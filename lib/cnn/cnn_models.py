from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3,  name="conv_test"),
            MaxPoolingLayer(2,2,'pooling'),
            flatten("flat"),
            fc(27,5, 0.02, "fc")
            ########## TODO: ##########
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=16,  padding=1,  name="conv_test1"),
            gelu(name="re1"),
            ConvLayer2D(input_channels=16, kernel_size=3, number_filters=32, stride=2, name="conv_test2"),
            gelu(name="re12"),
            MaxPoolingLayer(2,2,'pooling'),
            flatten("flat"),
            fc(1568,100, 5e-2, "fc1"),
            gelu(name="re13"),
            fc(100,20, 5e-2, "fc2")
            ########## TODO: ##########
            ########### END ###########
        )