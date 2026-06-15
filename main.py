###############################################
#   THIS IS NOT WORKING CODE,                 #
#   PURLY ANNOTATION FOR UNDERSTANDING        #
#   IN DEPTH LEARNING                         #
###############################################
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
        
#More control over your model---------------------------------------------------------------
import torch.nn.functional as F

# in SNN you have layers to computers thinking process... 
# - look for basic input data like image pixles 
# - multiply the data by its weights to look for 
#   specific patterns
#- sends the modified version of the data to the next layer
# a single layer does all of this but these are the steps to make
# AI do it's decision making, by creating more detailed 
# steps for it to follow you can have a more smarter AI


#define Network 

class Net(nn.Module):#creates class 
    def __init__(self): #runs automaticaly when create instance of this network
        super().__init__() #initializes PyTorch class so all mechanics work correctly
        
        #network dimentions ---------
        num_inputs=784 #number of inputs
        num_hidden= 300 #number of hidden neurons
        num_outputs= 10 #number of classes (ie, output neurons) for 0-9
        
        #decay rates--------
        # potential decaus over time if neuron doesn't spike. (beta)
        beta1=0.9 # sets global decay rate for all leaky neurons in layer 1
        beta2= torch.rand((num_outputs), dtype=torch.float)
            # independent decay rate for earch leay neuron in layer 2: [0,1)]
            #meaning a random starting decay betweenb 0 and 1 for each of 10 ouputs neurons
        
        #Defining Layers --------
        #sets up struucture and parameters of network
        self.fc1=nn.Linear(num_inputs,num_hidden)
            #creates fully connected (linear)layer connets 784 inputs to 300 hidden neurons
        self.lif1= snn.Leaky(beta=beta1) # not a learnable decay rate
            #creates spiking mechanism for hidden layer using beta1 decay
        self.fc2= nn.Linear(num_hidden,num_outputs)
            #connects the hidden layer to output layer
        self.lif2= snn.Leaky(beta=beta2, learn_beta=True)#learnable decay rate
            #creates spiking mechanism for output layer, dynamically 
            # adjust and train these decays over time
        
    def forward(self,x):# incoming data is x
        # this section defines how data flows through network over multiple steps
        mem1= self.lif1.init_leaky() 
            # reset/init hidden states at t=0
            #initializees internal voltage of hidden layer neurons to zero
        mem2= self.lif2.init_leaky() 
            # reset/init hidden states at t=0
            # initializes the membrane potential of output layer neurons to zero
        spk2_rec=[] 
            #record output spikes 
            #creates empy list to record output layer over time
        mem2_rec=[] 
            #record ouput hidden states
            #empty list to record how output layers membrade pot. changes over time
        
        #loop over time--------
        for step in range(num_steps): 
            #loops through data one time-step at a time
            
            #Processes 1st layer--------
            cur1=self.fc1(x.flatten(1))
                #flattens input data into 1D vector and passes it 
                # through first linear layer to calculate input 
                # current (cur1) hittin hidden layer
            spk1, mem1= self.lif1(cur1,mem1)
                #updates hidden layer. 
                #takes current and pervious membrain potential 
                #to compute new membrane pot.(mem1) to determne if any
                #nurons crossed pot. to create a spike 
                
            #processes 2nd layer--------
            cur2=self.fc2(spk1)
            spk2, mem1= self.lif1(cur2,mem2)
            
            spk2_rec.append(spk2) #saves output spikes from specifit time step into history list 
            mem2_rec.append(mem2) #saves output membrane pot from specifit time step into history list
            
        return torch.stack(spk2_rec), torch.stack(mem2_rec)
        #this return staks recorded list into orginized Pytorch (tensor)
        # and returns full history of output spikes and membrane pot. across all time steps
    
#load network onto cuda if avalible 
net=Net().to(device)
#----------------------------------------------------------
#this section creates mathmatical engine that will update network    
optimizer= torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9,0.999))
    #creates Adam optimizer (popular algrithm) 
    # used to update networks weights  net.parameters() it needs to tune
    #sets learning rate (how much it updates) lr=2e-3
    
    
loss_fn= SF.mse_count_loss(correct_rate=0.8,incorrect_rate=0.2)
    #defines mathrule (loss function) this specific one is 
    # Mean Square Error Spike Count Loss
    #this helps the AI determine how its predictions are wrong
    # and what to do to minimize it
    # the function uses correct_rate and Incorrect rate ratios to set targets
    # across T time steps. 
    # for T_correct = 0.8*T and T_incorrect=0.2*T
    #sums the actual spikes records binary pike output (0,1) for each nuron at every T and sums them up 
    # for the penalty score it is..
    # L= (1/c) sum from c=1 to C of (Target c- Actual c)^2

num_epochs =1 
    #run for 1 epoch - each data sample is seen only once
    #looks through entire dataset once and thats it
num_steps=25 
    #run 25 steps 
    #defines each image or sample shown for 25 time steos 

loss_hist=[] 
    #record loss over iterations 

acc_hist=[]
    #records accuracy over iterations

#more indepth training loop--------------------------------
for epoch in range(num_epochs):
    #loop for current epoch bv outer loop only runs once 
    
    for i, (data, targets) in enumerate (iter(train_loader)):
        #grabs batches of images (data) and correct lables(targets) one by one 
        #from dataset (train_loader) and i keeps track of which batch number we're on
        
        data= data.to(device)
            #sends batch to CPU or GPU depending on what your network is using
        
        targets= targets.to(device)
            #sends correct lables for images to same hardware
        
        net.train()
            # tells Pytorch to put network into training mode needed to learn. 
        spk_rec, _= net(data) 
            #forward-pass
            #passes images through network to get the output and saves output 
            # spikes and discards second ouput (mem pot) using the _
        loss_val = loss_fn(spk_rec,targets) 
            # loss calculation 
        optimizer.zero_grad() 
            #null gradients 
            #clears out memory of old calculations from privous batch so
            #they don't mess up the new math
        loss_val.backward()
            #calculate gradients 
            # how much each weight in network contributed to error
        optimizer.step() 
            #update weights
            #tweaks all weights slightly to make network more accurate next time
        loss_hist.append(loss_val.item()) 
            #store loss 
        
        # print every 25 iterations--------
        if i%25==0:
            net.eval()
            print(f"Epoch{epoch},Itterations{i} \n Train Loss: {loss_val.item(): .2f}")
            
            #check accuracy on a single branch---------
            acc= SF.accuracy_rate(spk_rec,targets)
                #counts spikes and calculates precentage was guessed right
            acc_hist.append(acc)
                #saves score
            print(f"Accuracy: {acc*100: .2F}%\n")
                #prints score