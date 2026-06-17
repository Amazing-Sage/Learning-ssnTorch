#########################################################################
# THIS IS TUTORIAL 2 FROM:                                              #
# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html   #
# Focuses on LIF neurons                                                #       
#########################################################################

#------------------------------------------------------------------------
#________________________________________________________________________
#Importing packages and setting up enviroments 
#________________________________________________________________________
#------------------------------------------------------------------------

import snntorch as snn 
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch 
import torch.nn as nn 

import numpy as np 

import matplotlib.pyplot as plt

import sys 
import os

# plot settings (copied& from another folder)

from config import plot_settings as ps

