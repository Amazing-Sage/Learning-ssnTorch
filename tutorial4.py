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

#========================================================================
#Synaptic Neuron Model in SNNTorch
#========================================================================
print("\nSynaptic Neuron Model in SNNTorch")
print("-----------------------------------\n")

# Temporal dynamics
alpha = 0.9 #decay rate of synaptic I
beta = 0.8 #decay rate of mempot
num_steps = 200

# Initialize 2nd-order LIF neuron
lif1 = snn.Synaptic(alpha=alpha, beta=beta)

# Periodic spiking input, spk_in = 0.2 V
w = 0.2
spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
spk_in = spk_period.repeat(20) #each weighted input I spike is sequencially passed in 

# Initialize hidden states and output
syn, mem = lif1.init_synaptic() 
    #syn is synaptic I at past time step 
    #mem is mem pot at past time step

spk_out = torch.zeros(1) #output spikes (1 if spike , 0 if none)
syn_rec = []#syn is synaptic I at current time step
mem_rec = []#mem is mem pot at current time step
spk_rec = []

# Simulate neurons
for step in range(num_steps):
  spk_out, syn, mem = lif1(spk_in[step], syn, mem)
  spk_rec.append(spk_out)
  syn_rec.append(syn)
  mem_rec.append(mem)


# convert lists to tensors-----------------------------------------------------------------------
#need to be torch.Tensor bc neuron modle has been 
# time shifted back 1 step w/out loss of generality

spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)

ps.plot_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec,
                     "Synaptic Conductance-based Neuron Model With Input Spikes")

#========================================================================
#Modeling Alpha Neuron Model
#========================================================================
print("\nModeling Alpha Neuron Model")
print("-----------------------------\n")

alpha = 0.8 #decay rate of positive exponential
beta = 0.7 #negitive decay rate of beta

#using this neuron is the same as the previous one, but the sum of 2 exp 
# functions requires synaptic I syn to be spit into syn_exc and syn_inh


# initialize neuron
lif2 = snn.Alpha(alpha=alpha, beta=beta, threshold=0.5)

# input spike: initial spike, and then period spiking
w = 0.85
spk_in = (torch.cat((torch.zeros(10), torch.ones(1), torch.zeros(89),
                     (torch.cat((torch.ones(1), torch.zeros(9)),0).repeat(10))), 0) * w).unsqueeze(1)
    #this creates a single sequence of spikes 
    #torch.zeros(10) is 10 step silence
    #torch.ones(2) is single spike at 11 
    #torch.zero(89) adds a 89 steps of cilence 
    #next does the same but spikes at step 1 and has9 steps of scilence, and repetes 10 times
    #using unsqueeze it formats all of this into 200 steps in a single 
    # line with 200 columns and multuplies by weight w(strength of spikes )
    #200 steps bc 1st torch.cat was 100 and second torch.cat was 100, cat these together to get 200

# initialize parameters
syn_exc, syn_inh, mem = lif2.init_alpha()
mem_rec = []
spk_rec = []

# run simulation
for step in range(num_steps):
  spk_out, syn_exc, syn_inh, mem = lif2(spk_in[step], syn_exc, syn_inh, mem)
  mem_rec.append(mem.squeeze(0))
  spk_rec.append(spk_out.squeeze(0))

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

ps.plot_cur_mem_spk(spk_in, mem_rec, spk_rec, "Alpha Neuron Model With Input Spikes")