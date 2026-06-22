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

# what's LIF? 
#takes the sum of weighted inputs and integrathe the input over time with 
#a leakage like a RC circuit. If integrated values exceeds threshold, the LIF 
#neuron will emit a voltage spike 

#overall, it's a way to remember spikes but erases memory after a sucessful spike

#========================================================================
#Lapicques LIF neuron model
#========================================================================

#time constant of 5E-3 s without stimulus
#this would be ideal if we want a neuron to relax to resting state and 
# provides a temporal memory mechanism by frogeting old imputs over time 
#like a RC circuit. 
#leakage stops membrane pot. from building spikes from past spikes
# and sets it close to resting state. 
#isolatied spikes fade out and sequence of rapid spikes build up to threashold
#it's also only spends energy when active spike is transmitted. 
num_steps=100
time_steps=1e-3
R=5
C=1e-3
U=0.9

def lif(U,time_step=1e-3,I=0,R=5e7,c=1e-10):
    tau=R*C
    U=U+(time_step/tau)*(-U+I*R)
    return U

U_trace=[] #record u for plotting 
for step in range(num_steps):
    U_trace.append(U)
    U=lif(U) # solve step of U

#LIF neuron, au 5e-3
lif1 = snn.Lapicque(R=R, C=C, time_step=time_steps)

#Inputs---
# cur_in is Iin passed as a input(0 for now)
# mem is the membrain pot. passed as input initialized U[0]=0.9V

#outputs---
#spk_out is output spike at the next time step. if 1 is a spike 0 no spike
#mem is membrain potential Umem at next time step 

#Initialize membrane, input, and output 
mem=torch.ones(1) * 0.9 #U=0.9 at t=0
cur_in= torch.zeros(num_steps,1) # I=0 for all t
spk_out= torch.zeros(1) # init. output spikes 

#list to store recording of membrane pot. 
mem_rec=[mem]

#run simulation at each time step to record mem and update mem_rec
#-----------------------------------------------------------------
print("Lapique:Step Input \n")
#pass updated value of mem and cur_in[step]=0 at every time step 
for step in range(num_steps):
    spk_out,mem=lif1(cur_in[step],mem)
    
    #store recordings of membrane potential 
    mem_rec.append(mem)
    
#convert the list of tensors into one tensor
mem_rec=torch.stack(mem_rec)
    
#pre-defined plotting function 
ps.plot_mem(mem_rec, "lapicque's Neurons Model W/out stimulus")

#initalize input current pulse 
cur_in=torch.cat((torch.zeros(10,1), torch.ones(190,1)*0.1),0)
    #input curent turns on at t=10

#initialize membrain and output and recordings 
mem=torch.zeros(1) #membrane pot of 0 at t=10 
spk_out=torch.zeros(1) #neurons need somewhere to sequentially dump its ouput spikes 
mem_rec= [mem] 

#new values of cur_in are passed to the neuron 
for step in range(num_steps):
    spk_out, mem=lif1(cur_in[step],mem) 
    mem_rec.append(mem)
    
#crunch - list of tensors in one tensor 
mem_rec= torch.stack(mem_rec) 
ps.plot_step_current_response(cur_in, mem_rec,10) 

#plot 
print(f"The calculated value of input pulse [A] x resistance [Ω] is: {cur_in[11]*lif1.R} V")
print(f"The simulated value of steady-state membrane potential is: {mem_rec[100][0]} V")


#========================================================================
#Pulse Input
#========================================================================
print("Lapcques: Pulse input\n")


