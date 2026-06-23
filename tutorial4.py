#########################################################################
# THIS IS TUTORIAL 4 FROM:                                              #
# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_4.html   #
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
import matplotlib.pyplot as plt

import sys 
import os

# plot settings (copied& from another folder)
from config import plot_settings as ps

