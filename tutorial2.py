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

#lapicques RC modle snntorch.Lapicqe 
#1st order modle snntorch.Leaky 
#synaptic Conducrtance-based neuron model snntorch.Synaptic


#------------------------------------------------------------------------
#________________________________________________________________________
#Lapicques LIF neuron model
#________________________________________________________________________
#------------------------------------------------------------------------

#trime constant of 5E-3 s

time_steps=1e-3
R=5
C=1e-3

#LIF neuron, au 5e-3
lif1=snn.Lapique(R=R, C=C, time_step=time_step0)

