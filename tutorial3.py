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
print("\ncode implimentation of simplified LIF neuron model")
print("--------------------------------------------------\n")
def lif(mem,x,w,beta,threshold=1):
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
    return spk, mem

# set neuronal parameters
delta_t = torch.tensor(1e-3) #time step of simulation
tau = torch.tensor(5e-3) #mem constant set to 0.005s decay about 37% og value if no input
beta = torch.exp(-delta_t/tau) #applies physicsl formula e^(delta_t/tau)
print(f"The decay rate is: {beta:.3f}")

#simulation matlab to check neuron behavior is correct to step V input 
#---------------------------------------------------------------------
print("\nsimulation matlab to check neuron behavior is correct to step V input")
print("---------------------------------------------------------------------\n")

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
  spk, mem = lif(mem, x[step], w=w, beta=beta)
  mem_rec.append(mem) # show mem pot slowly rise/ decay on graph
  spk_rec.append(spk) #when it spikes count on graph 

# convert lists to tensors shows lines of only spikes
mem_rec = torch.stack(mem_rec) #take mem potential add it to plot over 200 steps(nothing)
spk_rec = torch.stack(spk_rec) #take spike pot and add to plot of 200 steps(lines)

#actualy shows the three plots
ps.plot_cur_mem_spk(x*w, mem_rec, spk_rec, thr_line=1,ylim_max2=0.5,title="LIF Neuron Model With Weighted Step Voltage")


#========================================================================
#Leaky neuron modle in snntorch
#========================================================================
print("\nLeaky neuron modle in snntorch")
print("------------------------------\n")

#this helps connections between neurons(PyTorch) and create neurons(SnnTorch)

#initialize all layers 1st ------------------------------------------------
#--------------------------------------------------------------------------
# layer parameters
num_inputs = 784 #1st layer
num_hidden = 1000 #2nd Layer
num_outputs = 10 #3rd layer
beta = 0.99 #time constant

# initialize layers
fc1 = nn.Linear(num_inputs, num_hidden) #pyTorch is nn, so this creates path
lif1 = snn.Leaky(beta=beta) #snnTorch is snn. so it creates neurons
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta)

#initialize hidden variables and output of each spiking nerons ------------
#--------------------------------------------------------------------------
#init_leaky() sets mem pot to 0
#init_lapicque() does the same but for that modle, make sure to match everything 
#               to this specific model if using

# Initialize hidden states
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

# record outputs
mem2_rec = []
spk1_rec = []
spk2_rec = []

#create input spike train to pass to network-------------------------------
#--------------------------------------------------------------------------
print("\ncreate input spike train to pass to network")
print("----------------------------------------------\n")

#time*batchsize*feature_dimensions
# here there are 200 time steps across 748 input neurons 
# bc we're processing in mini patches we multuply by batch_size
#to "unsqueeze" the input along dim=1 to indicate 1 batch of data, 
# therefore 200*1*748
spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
print(f"Dimensions of spk_in: {spk_in.size()}")

print("\nloop simulates 2 layer SNN through time")
print("-----------------------------------------\n")
#loop simulates 2 layer SNN through time 
#bc we used torch.stack() we can loop through time line frame by frame
# network simulation
for step in range(num_steps):
    cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
    spk1, mem1 = lif1(cur1, mem1) 
        #spk_in[step] grabs input spikes for just specific milsec out of time line 
        #fc1() first set of weights to calc I (cur1) heading to 1st neuron 
        #lif1(cur1,mem1) first LIF takes new I and adds it to mem1, applies leak and checks for spikes 
        #if spike it it records spiked in spk1 and updates mem1 level 
        
    cur2 = fc2(spk1)
        #fc2(spk1) takes any spikes that passed layer 1 (sk1) and multuplies them by the 
        # second weight and turns it into a new I (cur2)
    spk2, mem2 = lif2(cur2, mem2)
        #lif2() is second layer of lif  and takes I and updates it own current mem pot(mem2)
        #and determines final output layer spikes (spk2)

    #records spikes and mem pot into lis#----------------------------------------------------------------------------t to look at later
    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

# convert lists to tensors (orginized timelines(tensors)to graph later)
mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)

#plot snn spikes 
ps.plot_spk_mem_spk(spk_in.squeeze(1).detach().cpu(),
                    spk1_rec.squeeze(1).detach().cpu(), 
                    spk2_rec.squeeze(1).detach().cpu(), 
                    "Fully Connected Spiking Neural Network")

#spike counter of the output layer-----------------------------------------
#--------------------------------------------------------------------------
print("\nspike counter of the output layer")
print("-----------------------------------\n")
from IPython.display import HTML

#visualizing results animated-------------------------------------------

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
spk2_rec = spk2_rec.squeeze(1).detach().cpu()
    #squeeze(1) removes empy dimention so plotting tool gets a clean gird of time steps 
    #detach() cuts tensor away from network learning history and tells us what was in the tensor 
    #cpu() network  running on GPU copies data to CPU bc it's required for plotting 
    

# plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

#  Plot spike count histogram
anim = splt.spike_count(spk2_rec, fig, ax, labels=labels, animate=True)
HTML(anim.to_html5_video())
# anim.save("spike_bar.mp4")

#visualizing results Traced---------------------------------------------

# plot membrane potential traces
splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
fig = plt.gcf()
fig.set_size_inches(8, 6)