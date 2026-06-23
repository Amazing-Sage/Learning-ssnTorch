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
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

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
