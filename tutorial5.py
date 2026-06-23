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
#Overcoming Dead Neuron Problem
#========================================================================
print("\nOvercoming Dead Neuron Problem")
print("--------------------------------\n")

# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
  def __init__(self, beta, threshold=1.0):
      super(LeakySurrogate, self).__init__()

      # initialize decay rate beta and threshold
      self.beta = beta
      self.threshold = threshold
      self.spike_gradient = self.ATan.apply

  # the forward function is called each time we call Leaky
  #updates membrane pot to see if it should spike
  def forward(self, input_, mem):
    spk = self.spike_gradient((mem-self.threshold))  
        # call the Heaviside function
        #checks if V crossed threshold if it did it output1
    reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
    mem = self.beta * mem + input_ - reset  
        # Eq (1), new input added and subtracts E if it just spiked (reset)
    return spk, mem

  # Forward pass: Heaviside function
  # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
  @staticmethod
  class ATan(torch.autograd.Function):
      @staticmethod
      def forward(ctx, mem):
          spk = (mem > 0).float() 
            # Heaviside on the forward pass: Eq(2)
            #sharp step function, souly 1 or 0 no half spikes 
          ctx.save_for_backward(mem)  
            # store the membrane for use in the backward pass
            #instead of sharp function, use equation (in backward def below)
            #to create a smooth curve over spike point
          return spk

      @staticmethod
      def backward(ctx, grad_output):
          (spk,) = ctx.saved_tensors  # retrieve the membrane potential
          grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
          return grad_output
      
# when neuron is built there's wo ways we can do this 

lif1 = LeakySurrogate(beta=0.9) #can be simulated using a forloop 

lif1 = snn.Leaky(beta=0.9) #the ATan surrogate gradient is applied to it by default 

#========================================================================
#Setting Up The Static MNIST DataSet
#========================================================================
print("\nSetting Up The Static MNIST DataSet")
print("-------------------------------------\n")

#same thing as the quick set up, we need this data set 
#so we can train it by importing images for it to use

# dataloader arguments
batch_size = 128
data_path='/tmp/data/mnist'

dtype = torch.float

 if torch.cuda.is_available(): 
    device= torch.device("cuda")
else:
    device=torch.device("cpu") # write cpu bc it's on windows not mac
    
# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


#========================================================================
#Defining Network
#========================================================================
print("\nDefining Network")
print("-------------------\n")

# Network Architecture
num_inputs = 28*28 # number of input neurons
num_hidden = 1000 # middle layer of 1k neurons to find patterns in images
num_outputs = 10 #output layer neurons

# Temporal Dynamics
num_steps = 25 #number of steps to un
beta = 0.95 #leak 5% with 95%retained of E charge from one time step 

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        #resets mem pot of all neurons to 0 to make sure old data doesn't effect new data
        mem1 = self.lif1.init_leaky() 
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        #time loop loops it 25 times to process data
        #at very end of each step spikes and mem pot of lif2 are appended to recording list
        #using torch.stack we append all of those together
        #cur is the input current 
        for step in range(num_steps):
            cur1 = self.fc1(x) 
                #fc1 applies linear transform to all input pixles from MNIST
            spk1, mem1 = self.lif1(cur1, mem1)
                #lif1 integrates weighted input over time emitting spike if condition met
                #if current strong enough over time mem1 will spike
            cur2 = self.fc2(spk1)
                #fc2 applies linear transformation to output spikes of lif1
                 
            spk2, mem2 = self.lif2(cur2, mem2)
                #lif2 is another spiking neuron layer integrating weited spikes over time 
                
            #these functions append spks and mempot in the end for final layer 
            # which is compiled in the return statement to be used later.
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Load the network onto CUDA if available
net = Net().to(device)

#========================================================================
#Training SNN
#========================================================================
print("\nTraining SNN")
print("-------------------\n")

#7.1 Accuracy metric ----------------------------------------------------
#------------------------------------------------------------------------

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target


def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))  
        #feeds data to output
        #data.view(batch_size,1) flattens input images into sibgle vector per images
        #net() returns recorded spikes and membrain potential 
    
    _, idx = output.sum(dim=0).max(1)
        #finding Preditctions 
        #output.sum(dim=0) bc output contains spikes recorded every timestep summing along dim=0 counts 
        #total spikes each output neuron over num_steps
        
    acc = np.mean((targets == idx).detach().cpu().numpy())
        #calculates accuracy 
        #idx is network guesses which is compared to actual ans targets 
        #calculates avg numbers and dent to cpu 

    #calls accuracy for both training and testing 
    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")
    
#7.2 Loss Definition ----------------------------------------------------
#------------------------------------------------------------------------

#nn.CrossEntropyLoss fucntion in PyTorch automatically handles taking 
#softmax of output layer as well as generating loss of output

loss=nn.CrossEntropyLoss() 

#7.3 Optimizer ----------------------------------------------------------
#------------------------------------------------------------------------

#learning rate is 5e-4
#Adam is a optimizer the preforms well on recurrent netowrks 

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9,0.999))

#7.4 One Iteration Of Training ------------------------------------------
#------------------------------------------------------------------------

#take first batch of data and load it onto CUDA if avalible 
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

#flatten input data to vector of size 784 and pass it through network 
spk_rec, mem_rec = net(data.view(batch_size, -1))
    #spk_rec records spks and records of mem into a vector
    
print(mem_rec.size()) #print recorded mem size

##torch.Size([25, 128, 10])
#recordings of membrane pot is taken across 25 time steps, 
#over 128 samples of data , 10 output neurons

#calculate loss at every time step and sum together 
# initialize the total loss value
loss_val = torch.zeros((1), dtype=dtype, device=device)

# sum loss at every step
for step in range(num_steps):
  loss_val += loss(mem_rec[step], targets)
  # for every step take the total loss val and 
  # add it to the loss of mem rec and targets 

print(f"Training loss: {loss_val.item():.3f}")
#Training loss: 60.488 # this is pretty bad bc it should be around 10%
#the reason for this is because currently the network is untrained. 

print_batch_accuracy(data, targets, train=True)
#Train set accuracy for a single minibatch: 10.16%

#single weight update is applied to the network 
# clear previously stored gradients
optimizer.zero_grad()

# calculate the gradients
loss_val.backward()

# weight update
optimizer.step()

#rerun loss calculation and accuracy after a single interatin
# calculate new network outputs using the same data
spk_rec,mem_rec=net(data_view(batch_size, -1))

#initialize total loss value 
loss_val=torch.zero((1),dtype=dtype, device=device)
#initialize loss value to store 1 number
# datatype determines whole or decimal 
# device tells it where to store it (CPU or GPU demending on what what set in the beginning)
#dtype is data type 

# sum loss at every step
for step in range(num_steps):
  loss_val += loss(mem_rec[step], targets)
  # for every step take the total loss val and 
  # add it to the loss of mem rec and targets 
  
print(f"Training loss: {loss_val.item():.3f}")
print_batch_accuracy(data, targets, train=True)

#Training loss: 47.384
#Train set accuracy for a single minibatch: 33.59%

#loss should decreace as accuracy increases 
#mem pot is used to calculate cross entropy loss
#spike count is used for mesure of accuracy 

