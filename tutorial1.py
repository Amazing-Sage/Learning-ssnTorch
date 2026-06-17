#########################################################################
# THIS IS TUTORIAL 1 FROM:                                              #
# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html   #
#                                                                       #         
#########################################################################

#------------------------------------------------------------------------
#________________________________________________________________________
#Importing packages and setting up enviroments 
#________________________________________________________________________
#------------------------------------------------------------------------

import snntorch as snn 
import torch 
from snntorch import spikegen

#training Parameters 
batch_size=128
data_path='/tmp/data/mnist'
num_classes=10 #MNIST has 10 output classes 

#torch Variables 
dtype = torch.float
 
#Download Dataset
#________________________________________________________________________
from torchvision import datasets, transforms

#define a transform to normalize the data 
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

mnist_train=datasets.MNIST(data_path,train=True, download=True, transform=transform)

#start traning networks, data_subset reduces the dataset by the factor defined in subset

from snntorch import utils 

subset=10
mnist_train=utils.data_subset(mnist_train, subset)
print(f"The size of mnist is {len(mnist_train)}")
#number should be 60k, this confirms the number of hadwritten digit images avalible to train machine

#creating dataloaders
#________________________________________________________________________
from torch.utils.data import DataLoader

train_loader=DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

#------------------------------------------------------------------------
#________________________________________________________________________
#spike Encoding 
#________________________________________________________________________
#------------------------------------------------------------------------
#mnist is not a time varring dataset but still spikes, 
#pass same training sample to network at each time step
#convert input into a spike train of sequence length, num_steps

#rate coding uses imput features to determine spiking frequency
#Do: Rate coding of MNIST (create a vector value of 0.5 and encode it)
#------------------------------------------------------------------------
#------------------------------------------------------------------------

#define target number 
images, targets = next(iter(train_loader)) #get the first batch of images and targets from the dataloader

#define output spikes 
output_spikes = spikegen.rate(images,num_steps=10) #use spikegen.rate because we're doing rate coding

#define predicted_lable to turn the predeicted number through spikes into a actual number 
predicted_label= output_spikes.sum(dim=0)[0].argmax().item()  

#piick first number of timages in that batch
first_target=targets[0]
    
print(f"Target number: {first_target.item()}")

#temporal Dynamics
num_steps=10 

#create vector filled with a 0.5
raw_vector= torch.ones(num_steps)*0.5
rate_coded_vector = spikegen.rate(raw_vector, num_steps=num_steps)
print(f"Converted vector:\n {rate_coded_vector}")

#tell us if the vector is spiking the right amount of times 
spike_counts=output_spikes.sum(dim=0)[0] #assuming output_spikes is tensor of spikes (time x batch x neurons)
#dim=0[0] tells us the sum of the spikes of the time dimention 
predicted_label= torch.argmax(spike_counts).item()

if first_target.item()==predicted_label:
    print(f"The predicted label{predicted_label} is correct.")
else:
    print(f"The predicted label {predicted_label} is incorrect.")

print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time")

#increase the vector length of raw_vector
num_steps=100 #this tells us that the number 

raw_vector= torch.ones(num_steps)*0.5
rate_coded_vector = spikegen.rate(raw_vector, num_steps=num_steps)
print(f"Increased vector length test complete.")
