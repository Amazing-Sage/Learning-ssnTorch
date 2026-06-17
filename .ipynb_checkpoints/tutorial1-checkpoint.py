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

#pick first number of timages in that batch
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

#------------------------------------------------------------------------
#________________________________________________________________________
#visualizing spike coding with matlab plots 
#________________________________________________________________________
#------------------------------------------------------------------------
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt #simplifies process of visualizing data
from IPython.display import HTML

#plot one sample of data into a single sample from the batch 
#dimention of spike_data
data_it, targets_it= mnist_train[0]
spike_data= spikegen.rate(data_it,num_steps=num_steps) #spike_data is a tensor of shape (time, batch, neurons)


spike_data_sample=spike_data.view(-1,28,28) # : is time steps, 0 is the first sample in the batch, 0 is the first neuron
print(spike_data_sample.size)

print("Corrected Shape:", spike_data_sample.shape)# debug, should see (num_steps,28,28)


fig,ax=plt.subplots()
anim=splt.animator(spike_data_sample,fig,ax)
HTML(anim.to_html5_video())

#target lable s
print(f" The corrisponding taret is: {targets_it}")

#MNIST grayscale images, 100% spiking every time sptep 
#to resuce the spiking frequency add the argument gain to reduce to 25%

spike_data2= spikegen.rate(data_it,num_steps=num_steps, gain=0.25)
spike_data_sample2= spike_data.view(-1,28,28)
fig,ax=plt.subplots()
anim2=splt.animator(spike_data_sample2,fig,ax)
HTML(anim.to_html5_video())

#average spikes out over time and recontruct imput images 

plt.figure(facecolor='w')
plt.subplot(1,2,1)
plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(),cmap='binary')
plt.axis('off')
plt.title('Gain =1')

plt.subplot(1,2,2)
plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(),cmap='binary')
plt.axis('off')
plt.title('Gain =0.25')

plt.savefig("mnist_spikes.png")
plt.show()  

