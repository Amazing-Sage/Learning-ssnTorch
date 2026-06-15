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
# this is part of loading essential tools needed to load and prepocess a dataset

from torch.utils.data import DataLoader
    #this takes large amount of data and splits it into small "bite sized data" to 
    #prevent crashing. it also shuggles data so network doesn't memorize order of the 
    #images it's trying to guess. 

from torchvision import datasets, transforms
    #torchvision.transforms  converts image into PyTorch numbers

    # "datasets" loads a large library of popular datasets to use as images and orginizes them
    #into folders.
    
    #"transforms" uses raw images from the internet in JPEG or PNG. 
    #However, it can't read JPEG so it will modigy the images converting 
    #them into math friendly numbers and adjusting the image so AI can read it 

#define a transform
    #this creates a list of modifications to every image for the AI to read
transform = transforms.Compose([
        transform.Resize((28,28)). #28 x 28 pixles
        transform.Greyscale(), #makes only black and white image
        transform.ToTensor(), #converts image to math
        transforms.Normalize((0,),(1,)) #squishes pixle values between math rage to help learn faster
    ])
        

#mnist dowloads data to computer
mnist_train= datasets.MNIST(data_path, train=True,download=True,transform=transform)
mnist_test=datasets.MNIST(data_path,train=False,download=True,transform=transform)

#create Dataloaders (feeds network images)
train_loader= DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_loader= DataLoader(mnist_test, batch_size=batch_size,shuffle=True)