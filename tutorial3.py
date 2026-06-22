#########################################################################
# THIS IS TUTORIAL 3 FROM:                                              #
# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html   #
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
#Code implimentation of Leaky Neuron Modle
#========================================================================
print("code implimentation of simplified LIF neuron model \n")
def lif(mem,x,w,beta,threashold=1):
#mem- membrane potential , acts nurons memory and how close to spiking 
#x - imput spikes/features tensor of binary spikes coming from 
#    previous layers of nurons or raw values if its from first layer
#w- weight, cales input z. when input spike (x=1) amount 
#   of I actually injected in neuron is x*w
#beta-  decay rate, how fast membrane potential decays back to its resting 
#       state if no input
#   if =1, nuron has perfect memory and never leaks 
#   if=0.9, neuron looses 10% charge ever time step
#threshold=1- if new calculated mem meets or exceeds val...
#              1. nuron ouput spike (1)
#              2. mem pot is reset to 0 or subtracted by threshold so it 
#                 doesn't spike for next frame.
#this all in general shows the functions... 
# Inew=x*w
# updated mem = (beta*mem)+Inew
#if updated mem >= threshold, then spike 1, if not reset to 0 

    spk = (mem > threshold) # if membrane exceeds threshold, spk=1, else, 0
    mem = beta * mem + w*x - spk*threshold 
        #leak= (beta*mem) if neuron frogets portion of past charge over time 
        #input= (x*w) incoming I (represented by x) scaled by connection strength (w)
        #reset= (-spk*threshold) if neuron filed spike, it subtracts threshold 
    return spk, me

# set neuronal parameters
delta_t = torch.tensor(1e-3) #time step of simulation
tau = torch.tensor(5e-3) #mem constant set to 0.005s decay about 37% og value if no input
beta = torch.exp(-delta_t/tau) #applies physicsl formula e^(delta_t/tau)
print(f"The decay rate is: {beta:.3f}")

#simulation matlab to check neuron behavior is correct to step V input 
#---------------------------------------------------------------------
print("simulation matlab to check neuron behavior is correct to step V input \n")

num_steps = 200

# initialize inputs/outputs + small step current input(current into netowrk)
x = torch.cat((torch.zeros(10), torch.ones(190)*0.5), 0)
    # creates tensor of 200 step current. 1st 10 steps input=0
    #at 10 steps, I is suddently switched on to constant val of 0.5 for remaning 190 steps
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron parameters
w = 0.4
beta = 0.819

# neuron simulation, shows spikes of mem total Umem (spike waves)
for step in range(num_steps):
  spk, mem = leaky_integrate_and_fire(mem, x[step], w=w, beta=beta)
  mem_rec.append(mem) # show mem pot slowly rise/ decay on graph
  spk_rec.append(spk) #when it spikes count on graph 

# convert lists to tensors shows lines of only spikes
mem_rec = torch.stack(mem_rec) #take mem potential add it to plot over 200 steps(nothing)
spk_rec = torch.stack(spk_rec) #take spike pot and add to plot of 200 steps(lines)

#actualy shows the three plots
ps.plot_cur_mem_spk(x*w, mem_rec, spk_rec, thr_line=1,ylim_max1=0.5,
                title="LIF Neuron Model With Weighted Step Voltage")