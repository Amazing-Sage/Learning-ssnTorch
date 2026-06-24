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

# plot settings (copied& from another folder)
from config import plot_settings as ps

#========================================================================
#Surgate Gradient Decent
#========================================================================
print("\nSurgate Gradient Decent")
print("--------------------------------\n")

# Leaky neuron model, overriding the backward pass with a custom function
## !! this is already in snn library and you don't need to code all this  !!
class LeakySigmoidSurrogate(nn.Module):
  def __init__(self, beta, threshold=1.0, k=25):

      # Leaky_Surrogate is defined in the previous tutorial and not used here
      super(Leaky_Surrogate, self).__init__()

      # initialize decay rate beta and threshold
      self.beta = beta
      self.threshold = threshold
      self.surrogate_func = self.FastSigmoid.apply

  # the forward function is called each time we call Leaky
  def forward(self, input_, mem):
    spk = self.surrogate_func((mem-self.threshold))  # call the Heaviside function
    reset = (spk - self.threshold).detach()
    mem = self.beta * mem + input_ - reset
    return spk, mem

  # Forward pass: Heaviside function
  # Backward pass: Override Dirac Delta with gradient of fast sigmoid
  @staticmethod
  class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, k=25):
        ctx.save_for_backward(mem) # store the membrane potential for use in the backward pass
        ctx.k = k
        out = (mem > 0).float() # Heaviside on the forward pass: Eq(1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors  # retrieve membrane potential
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.k * torch.abs(mem) + 1.0) ** 2  # gradient of fast sigmoid on backward pass: Eq(4)
        return grad, None

#!! use this to refrence the above code from snn library !!

#This all can be condensed by using the built in modual snn.surrogate from snntorch
#where k modulates how smooth the surrogate function is and treated like a hyperparameter 
# as k  inc the aprox converges twords og derive in og function (in tutorial) is dentotes as slope 
# surrogate gradient is passed into spike_grad as argyument 

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

#========================================================================
#Setting Up The CSNN
#========================================================================
print("\nSetting Up The CSNN")
print("--------------------------------\n")

# dataloader arguments
batch_size = 128
data_path='/tmp/data/mnist'

dtype = torch.float

#force device to use cpu 
device = torch.device("cpu")

# Define a transform
transform = transforms.Compose([ 
            transforms.Resize((28, 28)),#force every sing image to be 28*28 pixles
            transforms.Grayscale(), #forces image to be black and white
            transforms.ToTensor(), #converts image into a pytorch tensor
            transforms.Normalize((0,), (1,))]) #standardizes data mathmaticaly where the mean and Standard Deviation is at 

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) 
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
#data_path tells putorch where to save files 
#download=true checks if already have data set and if not it downloads it from the internet 
#transform=transform moment image pulled from storage it gets resozed, grayscaled, turned to tensor 
#train=true seterates data used for tetwork to practes on 
#train=false testing set is locked away to evaluate network later


# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
#drop_last=True total num of images doesn't divide by 128 perfectly, the tiny batch left 
# over if this doesn't happen will be thrown away, this is done do every single step always processes 128 images 

#2.2 Defining Networ ----------------------------------------------------
#------------------------------------------------------------------------

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
  #creates surrogate function. shart thresholds for spikes making
  # it zero everywhere except threshold
  #smooth aprox during backward pass so networks can calculate gradients and learn
beta = 0.5
num_steps = 50

# Define Network
class Net(nn.Module):
  def __init__(self):#network initialization
    super().__init__()

    # Initialize layers
    self.conv1 = nn.Conv2d(1, 12, 5)
      #2D convolutional laer takes a 1 channel image and outputs 12 features maps using a 5*5
    self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
      #1st layer of LIF neurons and recives output from conv1
    self.conv2 = nn.Conv2d(12, 64, 5)
    self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    
    #code below fully comnected layer that flattens 64*4*4 features 
    #into a cector and maps to be 10 output classes (0-9)
    #final layer spiking neurons that outputs actual classification spikes 
    self.fc1 = nn.Linear(64*4*4, 10)
    self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

  def forward(self, x):
    # Initialize hidden states and outputs at t=0
    mem1 = self.lif1.init_leaky()
    mem2 = self.lif2.init_leaky()
    mem3 = self.lif3.init_leaky()

    cur1 = F.max_pool2d(self.conv1(x), 2)
    #cur1 passes ijnput x through first convolution
    # and applies a 2*2 max pooling operation to reduce size
    #this becomes input current1 to neurons 
    spk1, mem1 = self.lif1(cur1, mem1)
    #updates neurons and takes I and previous mem pot and calc. 
    # new mem pot and output binary spikes

    #both of these below processes spikes from 1st layer to 
    #2nd conv and pooling layer then feeds them into 2nd spiking layer
    cur2 = F.max_pool2d(self.conv2(spk1), 2)
    spk2, mem2 = self.lif2(cur2, mem2)

    
    cur3 = self.fc1(spk2.view(batch_size, -1))
    #spk2.view(batch_size, -1) flattens spike tensor fron 2nd layer so can fit into fully connected layer
    spk3, mem3 = self.lif3(cur3, mem3)


    return spk3, mem3 #outputs final layers of likes and mem pot for this time step. 
  
#!!an alternitive way to do this !!
  
#  Initialize Network
net = nn.Sequential(nn.Conv2d(1, 12, 5), 
                    #only output spikes are sequentialy passed through layer wraped in nn.Sequential
                    
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    #init_hidden initializes hidden states of neuron( here mem pot)
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)
  
#2.3 Forward pass -------------------------------------------------------
#------------------------------------------------------------------------

#forward pass across simulation during num_steps:...

data, targets = next(iter(train_loader)) #read the data and targets
data = data.to(device) #send data to device
targets = targets.to(device) #send target data to device 

for step in range(num_steps):# for each step
    spk_out, mem_out = net(data) #reinitializes all mem pot to zero
    
#forward pass function 
def forward_pass(net, num_steps, data): 
  #reusable function that automates running network over time and collects outputs
  
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data) 
        #runs data through network for current time step and extracts output spikes and V
      spk_rec.append(spk_out)
        #saves current time step spikes into spike history list
      mem_rec.append(mem_out)
        #current steps V in V history list 

  return torch.stack(spk_rec), torch.stack(mem_rec) #stacks all indv time step tensors to one big tensor

spk_rec, mem_rec = forward_pass(net, num_steps, data)
  #call the function and runs all 50 times in this cas and saves complete history into two variables 
  
#========================================================================
#Training Loop
#========================================================================

#3.1 Loss using snn.Functional ------------------------------------------
#------------------------------------------------------------------------

#recordings of spike are passed as 1st argument to loss_fn and targen 
#neurons index as the 2nd argument to generate a loss

loss_val = loss_fn(spk_rec, targets)

print(f"The loss from an untrained network is {loss_val.item():.3f}")

#3.2 Accuracy using snn.functional --------------------------------------
#------------------------------------------------------------------------
acc = SF.accuracy_rate(spk_rec, targets)

#returns accuracy of single batch of data 
print(f"The accuracy of a single batch using an untrained network is {acc*100:.3f}%")

#returns accuracy on entire dataloader object
def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    #disables gradient calculation 
    #do this because evaluating modle and not training & saves memory and speeds up comp. 
    
    total = 0
    acc = 0
    net.eval()
    #sets network to evaluation mode

    train_loader = iter(train_loader)
    #converts trainloader to iterator so it can be looped step by step 
    
    for data, targets in train_loader: 
      #pulls one batch of images(data) and their correct lables(targes) at a
      #time untill whole dataset is processed.
      
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        #SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        #this determines precentage of correct preditions in this batch then 
        #multupl;ies it precentage by batch size and get abs num of correct images and add that to acc total
      
      total += spk_rec.size(1) #adds num of images to current batch to grand total

  return acc/total

test_acc = batch_accuracy(test_loader, net, num_steps)

print(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")

#3.3 Training Loop ------------------------------------------------------
#------------------------------------------------------------------------

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
  #initializes atam optimizer and tracts networks gradients (net.parameters())
  # with a learning rate (lr) or 0.01 to update weights smoothly

num_epochs = 1
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs): #tracks current training epoch

    # Training loop
    for data, targets in iter(train_loader): 
      #one batch of training images and correct lables
      
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train() #network training mode
        spk_rec, _ = forward_pass(net, num_steps, data)#runs batch through custom forward pass
        
        # initialize the loss & sum over time(how wrong it was)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()#clears out old graidents form previous batch so no accumulation
        loss_val.backward()#backpropigation calc. gradient of loss wrt every weight in network
        optimizer.step()#use clac. gradients to adjust networks weights

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        if counter % 50 == 0:
        with torch.no_grad():#stops tracting gradients and evaluatis
            net.eval()#evaluation mode

            # Test set forward pass
            test_acc = batch_accuracy(test_loader, net, num_steps)
              #calculates networks accuracy over entire dataset
              
            print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
            test_acc_hist.append(test_acc.item()) 

        counter += 1
