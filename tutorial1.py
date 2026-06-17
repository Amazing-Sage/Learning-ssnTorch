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
 


