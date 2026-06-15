#Import torch 
import torch, torch.nn as nn
import snntorch as snn

#Define variables for data loading
batch_size=128
    #batch_size is refrence to how many pictures per batch the processor will look at 
    #over the 60k pictures of numbers being used, it then is sent to the GPU
    #where the SNN looks at the 128 pictures and makes a guess what number is on that picture
    #After that it makes a mistake score and adjust the internal settings based on the 
    #mistake score and throws away those pictures. 
    # its the same as training the brain over and over to remember something and sharpen your 
    #-skills. 
data_path= '/tmp/data/mnist'
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #alternate way of writing this 
    #   if torch.cuda.is_available(): 
    #       device= torch.device("cuda")
    #   else:
    #       device=torch.device("cpu") // write cpu bc it's on windows not mac
       
        #what does this block of code do?
        # - torch.cuda.is_available() accesses Nvidas hardwear platform that allows 
        #   software to use their GPU for parallel processing, the true or false 
        #   checks if the Navidas graphics card with CUDA drivers are installed

#load MNIST darasheet 
from torch.utils.data import DataLoader
from torchvision import datasheets, transforms