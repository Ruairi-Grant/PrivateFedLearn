import os
import copy
import time
import pickle
import numpy as np
import torch

from torchsummary import summary
import torch.nn as nn

#rom options import args_parser
#from update_s3 import LocalUpdate
#from utils import test_inference
#from models import CNNMnistRelu, CNNMnistTanh
#from models import CNNFashion_MnistRelu, CNNFashion_MnistTanh
#from models import CNNCifar10Relu, CNNCifar10Tanh
#from utils import average_weights, exp_details
#from datasets import get_dataset
from torchvision import models
#from logging_results import logging

#from opacus.dp_model_inspector import DPModelInspector
#from opacus.utils import module_modification
#from opacus import PrivacyEngine

global_model = models.squeezenet1_1(pretrained=True)           
# for param in global_model.parameters():
#     param.requires_grad = False
global_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
global_model.num_classes = 5
summary(global_model, input_size=(3, 512, 512))