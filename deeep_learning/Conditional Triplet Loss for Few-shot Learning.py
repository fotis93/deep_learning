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


cudnn.benchmark = True


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0,a=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = a
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        ###worst triplets
        if distance_negative>distance_negative+self.margin:
            losses = losses +self.alpha* (distance_positive+distance_negative)/2
        ##best cases
        k = 0.99
        d = distance_positive - distance_negative
        if d>self.margin*(k-1) and d< self.margin*(2*k-1): 
            losses = losses -self.alpha* (distance_positive-distance_negative)/2

        return losses.mean()


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
   
def random_training(batch,newmodel):
    #creating a dummy copy of my data to operate upon(maybe not neccessary but i am not taking any chances )
    with torch.no_grad():

        nbatch = copy.deepcopy(batch)

        ones = [nbatch[0][i]for i in range(len(batch[0])) if nbatch[1][i].item()==1 ]
        zeros = [nbatch[0][i]for i in range(len(batch[0])) if nbatch[1][i].item()==0 ]

        one_len = (len(ones))
        zero_len = (len(zeros))

        anchors = []
        positives = []
        negatives = []

        #random creation of triplets 
        
        length = one_len+zero_len

        ###for zero class
        for i in range(length):
            anchor = ones[random.randint(0, one_len-1)]
            anchors.append(anchor)
            positive = ones[random.randint(0, one_len-1)]
            positives.append(positive)
            negative = zeros[random.randint(0, zero_len-1)]
            negatives.append(negative)

        ###for one class
        for i in range(length):
            anchor = zeros[random.randint(0, zero_len-1)]
            anchors.append(anchor)
            positive = zeros[random.randint(0, zero_len-1)]
            positives.append(positive)
            negative = ones[random.randint(0, one_len-1)]
            negatives.append(negative) 


        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        #print(anchors.size())
        #print(positives.size())
        #print(negatives.size())
        

    return (anchors,positives,negatives)



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


def train(model, criterion, optimizer,num_epochs=25):
    model.train()
    ###using the tripletloss loss function 
    for epoch in range(num_epochs):
        running_loss = []
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        #size_of_data = len(image_datasets['train'])
        #step = int(size_of_data/10)


        for batch  in dataloaders:
                #print(data)

                ##########print(len(collate(data)))

                batch = random_training(batch,Mode)
                #print(batch)
                
                #newmodel = copy.deepcopy(model).requires_grad_(False)


                #print(data)
                if batch == "next":
                    pass
                else:
                    #print(type(data))
                    inputs = []
                    for dato in batch:
                        dato = dato.to(device)
                        inputs.append(dato)

                    # zero the parameter gradients
                    #optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        #print(len(inputs[2][0]))
                        #print(len(inputs[1][0]))
                        #print(len(inputs[0][0]))


                        # inputs[0] is the anchor tensor,inputs[0] is the positives tensor,inputs[2] is the semiihard negatives tensor
                        for i in range(len(inputs[0])):
                            #
                            anchors = model(inputs[0][i].unsqueeze(0))
                            positives = model(inputs[1][i].unsqueeze(0))
                            hard_negatives = model(inputs[2][i].unsqueeze(0))

                            '''print(anchors)
                            print(positives)
                            print(hard_negatives)
                            print(fotis)'''
                            #print(outputs)                        
                            loss= criterion(anchors,positives,hard_negatives)
                            loss.backward()
                            optimizer.step()
                            #scheduler.step()
                            optimizer.zero_grad()
                            #print(loss)
                            running_loss.append(loss.cpu().detach().numpy())
                        #for param in model.parameters():
                            #print(param.data)
                            
                        #print(loss)
        print("running loss")
        print(np.mean(running_loss))
        print("epoch accuracy on train set")
        test(model,1,image_datasets['train'])
        print("epoch accuracy on test set")
        test(model,1,image_datasets['val'])

        #if epoch <25 or epoch >49:
        #    torch.save(model,f'deeep_learning/triplet_random_models/model_{epoch}')



    return model

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
    print("VAL")
    print(True_accepts)

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

    print("FAR")
    print(False_accepts)
    print()
    









    #print(fotis)



dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device ="cpu"
print(device)


#########setting up the network


# Parameters of newly constructed modules have requires_grad=True by default
Mode = SiameseNetwork()
Mode = Mode.to(device)
criterion=  torch.jit.script(TripletLoss())
optimizer_conv = optim.Adam(Mode.parameters(), lr=0.00005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


model2 = train(Mode, criterion, optimizer_conv,
                          num_epochs=75)



