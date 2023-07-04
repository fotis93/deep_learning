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

        if distance_positive >distance_negative+self.alpha:
            losses = losses+ self.alpha * (distance_positive+distance_negative)/2
        
        diff = distance_positive - distance_negative

        if diff<0 and diff > -self.margin:
            losses = losses +self.alpha* (distance_positive-distance_negative)/2

        if diff < self.margin and diff > -2*self.margin:
            losses = losses +self.alpha* (distance_positive-distance_negative)/2

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
   
def online_training(batch,newmodel):
    #creating a dummy copy of my data to operate upon(maybe not neccessary but i am not taking any chances )
    with torch.no_grad():

        nbatch = copy.deepcopy(batch)

        ones = [nbatch[0][i]for i in range(len(batch[0])) if nbatch[1][i].item()==1 ]
        zeros = [nbatch[0][i]for i in range(len(batch[0])) if nbatch[1][i].item()==0 ]

        one_len = (len(ones))
        zero_len = (len(zeros))

        if one_len <2 or zero_len <2 :
            return "next"
        

        images_1=torch.stack(ones)
        images_0=torch.stack(zeros)

        images_11=copy.deepcopy(images_1).detach()
        images_01=copy.deepcopy(images_0).detach()

        #creating the embeddings for the images 
        ###adding the [0] for the list return of my custom model

        newmodel1 = copy.deepcopy(newmodel).requires_grad_(False)   

        
        #print(images_11.size())
        embedding_0 = newmodel1(images_01).detach()
        embedding_1 = newmodel1(images_11).detach()
        #print(embedding_1.size())
        #print(embedding_0.size())

        #print(fotis)

        mean0, std0 = torch.mean(embedding_0), torch.std(embedding_0)
        mean1, std1 = torch.mean(embedding_1), torch.std(embedding_1)

        embedding_0=(embedding_0-mean0)/std0
        embedding_1=(embedding_1-mean1)/std1
        
        ##using [0] because it is a tensor in a tensor 
        zero_anchor_triplets = (triplet_creation(embedding_0,embedding_1))
        #print()
        one_anchor_triplets = (triplet_creation(embedding_1,embedding_0))

        anchors = []
        positives = []
        negatives = []

        #print(zero_anchor_triplets)

        ###appending the zero anchor triplets 
        # for class zero
        for triplet in zero_anchor_triplets:
            '''print(triplet)
            print("zeros")
            print(len(zeros))
            print(zeros)
            print("ones")
            print(ones)
            print(len(ones))'''

            #print(zeros)
            #print(triplet[0])
            anchors.append(zeros[triplet[0]])
            positives.append(zeros[triplet[1]])
            negatives.append(ones [triplet[2]])

        ###appending the one anchor triplets 
        ## for class one
        for triplet in one_anchor_triplets:
            '''print(triplet)
            print("zeros")
            print(len(zeros))
            print(zeros)
            print("ones")
            print(ones)
            print(len(ones))'''
            anchors.append(ones[triplet[0]])
            positives.append(ones[triplet[1]])
            negatives.append(zeros [triplet[2]])

        
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        #print(anchors.size())
        #print(positives.size())
        #print(negatives.size())
        

    return (anchors,positives,negatives)

def triplet_creation(emb1,emb0):
    triplets = []
    for i in range(len(emb1)):
        anchor = emb1[i]
        #print(anchor)

        pos_matrix = emb1-anchor
        pos_dist = torch.matmul(pos_matrix, pos_matrix.t())
        pos_sq_dist=pos_dist.diagonal()
        ###location of hard positive 
        hard_pos = torch.argmax(pos_sq_dist)
        hp = torch.max(pos_sq_dist)

        neg_matrix = emb0 - anchor 
        neg_dist = torch.matmul(neg_matrix, neg_matrix.t())
        neg_sq_dist=neg_dist.diagonal()
        ####checking for semihard negatives if they do not exist i will use the hard negatives 
        semi_hard_lis = (neg_sq_dist>hp)
        semi_hard_lis=semi_hard_lis.tolist()

        index_list = range(len(semi_hard_lis))
        mask = semi_hard_lis
        semi_h = list(compress(index_list, mask))
        '''print(index_list)
        print(mask)
        print(semi_h)'''
        if len(semi_h)==0:
            hard_neg = torch.argmin(neg_sq_dist)
            triplet=[i,hard_pos.item(),hard_neg.item()]
            triplets.append(triplet)
        else:
            for j in semi_h:
                semi_hard_neg = j
                triplet=[i,hard_pos.item(),semi_hard_neg]
                triplets.append(triplet)



        #for item in table :


        #print(fotis)
        ####.tem to get the number out of the tensor
        #triplet=[i,hard_pos.item(),hard_neg.item()]
        #triplets.append(triplet)
    return(triplets)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

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


def train(model, criterion, optimizer,num_epochs=60):
    model.train()
    loss_list=[]

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

                batch = online_training(batch,Mode)
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
        loss_list.append(np.mean(running_loss))
        print("running loss")
        print(np.mean(running_loss))
        print("epoch accuracy on train set")
        test(model,1,image_datasets['train'])
        if epoch>=50:
            torch.save(model,f'deeep_learning/triplet_selection_models/model_{epoch}')



    return model,loss_list

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
Mode.apply(init_weights)
Mode = Mode.to(device)
criterion=  torch.jit.script(TripletLoss())
optimizer_conv = optim.Adam(Mode.parameters(), lr=0.00005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


model2 = train(Mode, criterion, optimizer_conv,
                          num_epochs=75)[0]

#print(offline_training(image_datasets['train'],4,Mode))

#print(len(image_datasets['train'])/20)


