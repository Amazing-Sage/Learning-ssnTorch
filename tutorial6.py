#########################################################################
# THIS IS TUTORIAL 6 FROM:                                              #
# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html   #
# Focuses on Feedforward Spiking Neural Network                         #       
#########################################################################

#========================================================================
#Importing packages and setting up enviroments 
#========================================================================

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import itertools

import sys 
import os

# plot settings (copied& from another folder)
from config import plot_settings as ps

#========================================================================
#Surgate Gradient Decent
#========================================================================
print("\nSurgate Gradient Decent")
print("--------------------------------\n")