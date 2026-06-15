#Import torch------------------------------------------------------------------------------
import torch, torch.nn as nn
import snntorch as snn

#Define variables for data loading---------------------------------------------------------
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

#load MNIST datasheet----------------------------------------------------------------------
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

#define a transform------------------------------------------------------------------------
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

#Define network with SNNTorch--------------------------------------------------------------

from snntorch import surrogate

beta = 0.9 #neuron decay rate
spike_grad = surrogate.fast_sigmoid() 

#Initialize Convlutional SNN 

net= nn.Sequential (nn.Conv2d(1,8,4),
                    nn.MaxPool2d(2),
                    snn.Leaky (beta=beta, spike_grad=spike_grad,init_hidden=True),
                    nn.Conv2d(8,16,5),
                    nn.MaxPool2d(2),
                    snn.Leaky (beta=beta, spike_grad=spike_grad,init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(16*4*4,10),
                    snn.Leaky (beta=beta, spike_grad=spike_grad,init_hidden=True, output=True)
                ). to(device)

    #what does this code mean?...
        # nn.Conv2d are the feature finders (convolutional Layers)
        #   - scans across image looking for specific patterns
        
        # nn.MazPool2d are max pooling layers (the shrinkers)
        #   - looks for at a small grid of pixles and only keeps the brightest pixle throwing away
        #     the rest
        
        # snn.Leaky(beta=beta,..) spiking brain cells 
        #   - these are the artifical neurons that make this a SNN 
        #     take information from the image and charge up over time like a battery 
        #     when when they reach the max they spit out a 1. If there isn't enough 
        #     information within that time, they get a 0 and reset. 
        
        # nn.Flatten() and nn.Linear (final decision maker)
        #   - flatten takes the 2D grid of beurons and puts them into a 1D single line of numbers
        #   - linear connects the line into 10 output neurons ( one for earch digit 0-9)
        
        # Final snnLeaky / .to(device)
        #   - snnleaky (...,output=True) monitiors those 10 output options over time. Whichever 
        #     number (0-9) spikes the most that's what the final guess will be 
        #     ie. most spikes at 3 means the guess will be 3
        #   - .to(device) pushes the entire brain onto the GPU so it can think faster
        
#Training Loop---------------------------------------------------------------------------
    #set neuron with highest firing rate and mesure accracy

num_epochs=1 #run for 1 epoch - each data sample is seen only onve 
num_steps=25 #run for 25 steps 

loss_hist = [] #record loss over iterations
acc_hist=[] #record accuracy over iterations

#training loop 
for epoch in range(num_epochs):
    for i, (data,targes) in enumerate(iter(train_loader)): #hands AI batch of images
        data = data.to(device)
        targets =targets.to(device) 
        
        net.train()
        spk_rec = forward_pass(net,data,num_steps) 
            # forward-pass (learning mode) and feeds image to AI to then 
            # flash images over multiple time steps and records how many times 
            # the output neurons spiked
        loss_val = loss_fn(spk_rec,targets) # loss calculation (compares guess to ans)
        optimizer.step() 
            #update weights reports outcome and adjust 
            # internal connections(weights) of networks 
            # if spiked at wrong time wit will weaken that connection
        loss_hist.append(loss_val.item()) #store loss (saves score into history list)
        
    #print every 25 iterations 
    if i% 25==0:
        print(f"Epoch {epoch}, Iteration{i} \n Trainloss:{loss_val.item():.2f}")
            #prints progress report every 25 batches
        
        #check accuracy on a single branch 
        acc = SF.accracy_rate(spk_rec,targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc*100:.2F}%\n")
        
    