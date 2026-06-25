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

import tonic

# plot settings (copied& from another folder)
from config import plot_settings as ps

#========================================================================
#Using Tonic to load neuromorphic Datasets
#========================================================================

#Tonic works alot like PyTorch cision. 
#loading neuromorphic version of MNIST dataset called NMNIST. THis can 
#show us some raw events to get a feel with what we're working with. 
dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
events, target = dataset[0]

print(events)

#should print this 
# [(10, 30, 937, 1) (33, 20, 1030, 1) (12, 27, 1052, 1) ...
# ( 7, 15, 302706, 1) (26, 11, 303852, 1) (11, 17, 305341, 1)]

#in the above answer, each row corresponds to a single event which cosist of..
#x,y,time stanp, polarity
#x,y corrispond to place on the grid
#time stamp is in microseconds 
#polarity is a +1 spike or a -1 spike accured (brightness inc or dec)

#if we where to accumulate those events over time and plot the bins as images it would look like this

tonic.utils.plot_event_grid(events)

#1.1 transformations ----------------------------------------------------
#------------------------------------------------------------------------

#neural nets don't take list of events as imputs, instead the data must me converted to a tensor
#we can choose a set of transforms to apply our data before feeding it to our network

#to bin events into smaller num or frames use ToFrame transformation#2.1 Defining network ---------------------------------------------------
#------------------------------------------------------------------------
#this reduces temporal precision and also alows to work with dense formats
import tonic.transforms as transforms

sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      #event filtered if no events
                                      #occur within neghbourhood of 1 pixle across event is filltered out denoting one off events
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=1000)#integrades events into 1k us bins
                                     ])

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

#1.2 Fast Dataloading ---------------------------------------------------
#------------------------------------------------------------------------

#og data is stored in format that's slow to read, so instead we use disk caching,
#this is when files are loaded from og dataset and then written to the disk

#use tonic.collation.PadTensors that pad out shorter recordings to ensure all batches have same dimentions

from torch.utils.data import DataLoader
    #imports Pytorch standard data deliveryy tool that manages grouping data into batches and shuffles them
from tonic import DiskCachedDataset
    #import special tool from Tonic that saves processed data to hard drive so computer doesn't have to recal. every time

cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    # takes og dataset (trainset) and applies all transformations defined earlier and saves 
    # them to read traight from this cache when it runs
cached_dataloader = DataLoader(cached_trainset)
    #creates basic loader that pulls exactly one single data sample at a time from new cached folder
batch_size = 128
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
    #creates main loader for training and froups 128 samples together. 
    # uses 
def load_sample_batched():
    #small helpter function specifically to tst how fast single item can be grabbed from cache
    events, target = next(iter(cached_dataloader))
        #grabs first avalible sample of event frames and corrisponding lable from loader
    
#if lots of ram space you can speed up dataloading further by caching main memory instead of dist 
#from tonic import MemoryCachedDataset
#cached_trainset = MemoryCachedDataset(trainset)

#========================================================================
#Training Network using frames created from events
#========================================================================
#train netowrk on the NMNIST classification task we start by defining caching wrappers and dataloaders
#also apply augentations to training data. bc these are frames we can use PyTorch cision to apply whatever transform we want


transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])
    #tonic.transforms.Compose- groups multiple data clearning steps together into a line so they can run in order
    #torch.from_numpy- converts raw data to arrays into PyTorch Tensors so computer can do math using GPU
    #torchvision.transforms.RandomRotation([-10,10])]- randomly tilts event image +- 10 deg everytime they're loaded(teach to reconize shapes not patterns)
    
cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')
    #saves training data from earlier and saves results to a folder on the hardrive

# no augmentations for the testset
cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')
#saves test data to diffrent folder. Don't add random rotations hear bc we want to test network on clean realistic data

batch_size = 128

#creates train and dtest loaders
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))
#dataloader groups datasets into batches 
#collate_rn tells dataloader how o pack list of indv sampleas into a single batch tensor 
# this is done by setting it to tonic.collation.PadTensors(batch_first=False)...
# this looks at all sequences in current batch and finds longest one 
#it then takes all shorter sequences and adds blocks of zeros to end untill they all match the length of th longest sequence 
#after this it's stacked into a neat datacube (tensor batch)


event_tensor, target = next(iter(trainloader))
#use to grab one batch of data, this is done through iter(trainloader) which creates a iterator out of dataloader 
#next(..) tells python to pull next item from that iterator that just opened, therefor pulling very first batch of data
#event_tensor, target unpacks event_tensor becomes a regular tensor containing event data and target has the tensor with matching labels.

# if we want the entire dataset steped through batch by batch,  we do a forloop : for event_tensor, target in trainloader 
print(event_tensor.shape)
#torch.Size([311, 128, 2, 34, 34]) --> this represents time,batch,channels(polariity tracked on 2 channels in this case),height,width

#2.1 Defining network ---------------------------------------------------
#------------------------------------------------------------------------

device = torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.atan() 
    #surrogate.atan() creates archtan surrogate gradient 
    # we do this for forward pass when the network still uses sharp 0 or 1 spikes (predition)
    # we also do this for backward pass (training) where it uses a arc tan curve 
    # to pretend the sharp spike was actualy a smooth curve to help update weights
beta = 0.5

#  Initialize Network
net = nn.Sequential(nn.Conv2d(2, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 32, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(32*5*5, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

# this time, we won't return membrane as we don't need it

def forward_pass(net, data):#(preditions part)
  spk_rec = []#record spikes in this list
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])#selects one specific msec for fram across batch and reduces it to shape ( batch, channels, height, width)
    
      spk_rec.append(spk_out)#adds that reduced shape of the spk out to the spk_rec list

  return torch.stack(spk_rec)#stack all spk_rec together

#2.2 Training -----------------------------------------------------------
#------------------------------------------------------------------------

#uses mean square error spike count loss, aims to elicit spikes from correct class 80% of the time and 20% from incorrect class
#encouraging inccorect neurons to fire could be motivated to avoid dead neurons 
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)


num_epochs = 1
num_iters = 50 #takes a long time so ths is a good one bc its 1/10th of a full epoch

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(trainloader)):
            # pulls single batch of data out of loader
            # data contains event tensors 
            #targets have correct labels 
            #enumerate counts batches for you saving the current batch number i 
            
        data = data.to(device)
        targets = targets.to(device)

        net.train()#training mode 
        spk_rec = forward_pass(net, data) #feeds batch into network and returns spike rec over time
        loss_val = loss_fn(spk_rec, targets) 
            #calculates error and compares how many time neurons 
            #fired for each class against actual targest and see how wrong it was

        # Gradient calculation + weight update
        optimizer.zero_grad() 
            #erases old gradients (gradients are math instructions for how to change weights) 
            # from previous batch, if you don't do this it will add to the next batch
            
        loss_val.backward()
            # back propication and cal how much each weight in network contributed to error 
            #this is wehre the surrogate.atan() is used
        optimizer.step()
            #takes calculated gradients and tweeks networks weights to be mo

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, targets)
        #helper function from snn torch that calculates precentage of batch and prints it
        acc_hist.append(acc)
        #adds ans from above to list 
        print(f"Accuracy: {acc * 100:.2f}%\n")

        # training loop breaks after 50 iterations
        if i == num_iters:
          break
      
#should print out something like this...!! takes a long time to run !!
    #Epoch 0, Iteration 0
    #Train Loss: 31.00
    #Accuracy: 10.16%

    #Epoch 0, Iteration 1
    #Train Loss: 30.58
    #Accuracy: 13.28%
    
#========================================================================
#Results (plotting)
#========================================================================

#3.1 plot test accuracy--------------------------------------------------
#------------------------------------------------------------------------
# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()


#3.1 plot test accuracy--------------------------------------------------
#------------------------------------------------------------------------
spk_rec = forward_pass(net, data)

#this runs a forward pass on a batch of data to obtain spike recordings 

#changing idx allows to index various samples from simulated mini batch
# use splt.spike_count to explore spiking behavior for a few diffrent samples 

from IPython.display import HTML

idx = 0

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
print(f"The target label is: {targets[idx]}")

# plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

#  Plot spike count histogram
anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels,
                        animate=True, interpolate=1)
    #interpolate=1
    #this dictates how smooth animation looks by adding extra frames betw actual network steps
    #setting it equal to 1 is the default. 
    
    
HTML(anim.to_html5_video())
# anim.save("spike_bar.mp4")

The target label is: 3

