#########################################################################
# THIS IS TUTORIAL 1 FROM:                                              #
# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html   #
#                                                                       #         
#########################################################################

#________________________________________________________________________
#Importing packages and setting up enviroments 
#________________________________________________________________________

import snntorch as snn 
import torch 

#training Parameters 
batch_size=128
data_path='/tmp/data/mnist'
num_classes=10 #MNIST has 10 output classes 

#torch Variables 
dtype = torch.float
 
#________________________________________________________________________
#Download Dataset
#________________________________________________________________________
from torchvision import datasets, transforms

#define a transform to normalize the data 
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Greyscale(),
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



