from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import itertools
from itertools import compress
import torch.nn.functional as F
import csv


cudnn.benchmark = True

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)

        #self.fc1 = nn.Linear(64 * 2 * 2, 120)
        #self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc1 = nn.Linear(64 * 15 * 15, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, arg):
        output = self.pool(F.relu(self.conv1(arg)))
        #print(output.size())
        output = self.pool(F.relu(self.conv2(output)))
        #print(output.size())
        output = self.pool(F.relu(self.conv3(output)))
        #print(output.size())
        output = self.pool(F.relu(self.conv4(output)))
        #print(output.size())
        output = torch.flatten(output, 1) # flatten all dimensions except batch
        #print(output.size())
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        return output
   

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(330),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir =  'deeep_learning/IDphotos'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}


dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=40,
                                             shuffle=True )

test_dataloaders = torch.utils.data.DataLoader(image_datasets['val'], len(image_datasets["val"]),
                                             shuffle=True )
custom = torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                             shuffle=True )  


############################### trainings the network ###########################################



def test (model,dist,dataset):
    model.eval()
    pos=[]
    neg=[]
    #loading the whole dataset and breaking it into out two classes 
    for item in dataset:
        if item[1]== 1:
            neg.append(item[0])
        else:
            pos.append(item[0])
    # making our list of tensors into a tensor 
    pos= torch.stack(pos)
    neg= torch.stack(neg)
    #embedding the images 
    pos_vec=model(pos).detach()
    neg_vec=model(neg).detach()

    #calculating the size of all the positives pairs 
    #size is the combination of positive vectors 
    pos_n=len(pos_vec)
    #Psame = int( np.math.factorial(pos_n)/(2*np.math.factorial(pos_n-2)) )
    #print("Psame")
    #print(Psame)
    
    #calculating the correct predictions of the classifier
    corrects = 0
    P_list = list(itertools.combinations(pos_vec, 2))
    #print("hi")
    Psame = (len(P_list))
    for item in P_list:
        vec = item[0]-item[1]
        pos_dist = torch.matmul(vec, vec.t())
        #print(pos_dist)
        if pos_dist<dist*dist:
            corrects+=1
    #this is the true accepts as in the paper 
    True_accepts = corrects/Psame
    #print("True_accepts")
    #print(True_accepts)


    # calculating the size of all the different pairs pos - neg 
    
    neg_n = len(neg_vec)
    #size of all the positives * size of all the negatives vectors 
    Pdiff = pos_n*neg_n
    # calculating false accepts
    false_ac=0
    for po in  pos_vec:
        for ne in neg_vec:
            vec = po-ne
            fa_pos_dist = torch.matmul(vec, vec.t())
            if fa_pos_dist<dist*dist:
                false_ac+=1
    False_accepts = false_ac/Pdiff

    #print("False_accepts")
    #print(False_accepts)
    #print()

    return True_accepts,False_accepts
    


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device ="cpu"
print(device)


#########setting up the network

Train = []
Test = []

# Parameters of newly constructed modules have requires_grad=True by default
for i in range(25):
    Mode = torch.load(f'deeep_learning/triplet_random_models/model_{i}')
    Mode = Mode.to(device)
    Mode.eval()
    count = 0.2

    print(f"model {i}")
    while count<= 1.7:
        Te = test(Mode,count,image_datasets['train'])
        Tr = test(Mode,count,image_datasets['val'])
        Telist = [i,count,Te[0],Te[1]]
        Trlist = [i,count,Tr[0],Tr[1]]
        Test.append(Telist)
        Train.append(Trlist)
        '''print("True_Accepts")
        print(Te[0],Tr[0])
        print("False_Accepts")
        print(Te[1],Tr[1])'''
        count+=0.5

fields =["Epoch","distance","VAL","FAR"]

with open('Test.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(fields)
    write.writerows(Test)


with open('Train.csv', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(fields)
    write.writerows(Train)
