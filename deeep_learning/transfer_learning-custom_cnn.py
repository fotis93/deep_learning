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
from sklearn import metrics

import torch.nn.functional as F


cudnn.benchmark = True
plt.ion()   # interactive mode

######################################################################
# custom CNN we created 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


######################################################################
# Load Data

##data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #in order to not learn the skin colour and open with it 
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = "deeep_learning/IDphotos"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



######################################################################
# Training the model

def train_model(vers,model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_fpr = 0
    best_fnr = 0
    best_epoch = 0
    best_precision = 0
    best_recall =0 

    ####lists to keep track of statistics 
    train_acc_list= []
    test_acc_list = []

    train_loss_list = []
    test_loss_list = []

    train_fpr_list = []
    test_fpr_lsit = []

    train_fnr_list = []
    test_fnr_list = []

    train_precision_list =[]
    test_precision_list =[]

    train_recall_list =[]
    test_recall_list =[]


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        tot_labels = []
        tot_prob = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            False_positives = 0
            True_negatives = 0

            False_negatives = 0
            True_positives = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    ###calculating probabilites 
                    probabilites =[]
                    for item in outputs:
                        probab = torch.nn.functional.softmax(item, dim=0)
                        probability = torch.max(probab)
                        probabilites.append(probability.item())


                    tot_labels=tot_labels +labels.tolist()
                    tot_prob=tot_prob + probabilites
                    '''print(preds)
                    print(tot_labels)
                    print(tot_prob)
                    print(fotis)'''
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                False_pos = [1 for i in range(len(preds)) if (preds[i]==1 and labels[i]==0) ]
                true_neg = [1 for i in range(len(preds)) if (preds[i]==0 and labels[i]==0) ]
                True_negatives +=sum(true_neg)
                False_positives +=sum(False_pos)

                False_neg = [1 for i in range(len(preds)) if (preds[i]==0 and labels[i]==1) ]
                True_pos = [1 for i in range(len(preds)) if (preds[i]==1 and labels[i]==1) ]
                False_negatives += sum(False_neg)
                True_positives += sum(True_pos)


            ##################################3
            ###calculating ROC ,PRC for each epoch 
            #####calculating roc curve 


            ########creating the curves for each epoch 
            fpr, tpr, thresholds = metrics.roc_curve(tot_labels, tot_prob)

            path = 'deeep_learning/'+"roc_curves_" + vers
            f_path = os.path.join(path , "epoch%d.jpg" % epoch)
            plt.figure(epoch+len(vers)*10000)
            plt.plot(fpr, tpr,label=f'{phase}',marker='.')
            plt.legend()
            plt.savefig(f_path, bbox_inches='tight')
            #plt.show()


            precision, recall, thresholds = metrics.precision_recall_curve(tot_labels, tot_prob)

            path = 'deeep_learning/'+"prc_curves_" + vers
            f_path = os.path.join(path , "epoch%d.jpg" % epoch)
            plt.figure(epoch+25+len(vers)*10000)
            plt.plot(precision,recall,label=f'{phase}',marker='.')
            plt.legend()
            plt.savefig(f_path, bbox_inches='tight')
            #plt.show()



                #running_ones = 
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            FPR = False_positives/(False_positives + True_negatives)
            FNR = False_negatives/(False_negatives + True_positives)
            precision = True_positives/(True_positives+False_positives)
            recall = True_positives/(True_positives+False_negatives)


            if phase == "train":
                    train_acc_list.append(epoch_acc.item())
                    train_loss_list.append(epoch_loss)
                    train_fpr_list.append(FPR)
                    train_fnr_list.append(FNR)
                    train_precision_list.append(precision)
                    train_recall_list.append(recall)
    

            else:
                    test_acc_list.append(epoch_acc.item())
                    test_loss_list.append(epoch_loss)
                    test_fpr_lsit.append(FPR)
                    test_fnr_list.append(FNR)
                    test_precision_list.append(precision)
                    test_recall_list.append(recall)



            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} FPR: {FPR:.4f} FNR: {FNR:.4f}  ')


            # if i do not have better model i want to break the loop
           
            ''' if phase == 'val'and breakcount == 5:
                best_model_wts = copy.deepcopy(model.state_dict())
                return model
            if phase == 'val':
                breakcount+=1'''
            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_fpr = FPR
                    best_fnr = FNR
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_precision = precision
                    best_recall =recall
                elif epoch_acc == best_acc and best_fpr > FPR:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_fpr = FPR
                    best_fnr = FNR
                    best_precision = precision
                    best_recall =recall
                    best_model_wts = copy.deepcopy(model.state_dict())

            '''if phase == 'val':
                print(breakcount)'''
            

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best epoch: {best_epoch}')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Best val FPR: {best_fpr:4f}')
    print(f'Best val FPR: {best_fnr:4f}')
    print(f'Best val precision: {best_precision:4f}')
    print(f'Best val recall: {best_recall:4f}')


    #######training graphs 
    path = 'deeep_learning/graphs'
    epochs = [i for i in range(25)]

    train_l = [train_acc_list,train_loss_list,train_fpr_list,train_fnr_list,train_precision_list,train_recall_list]
    test_l = [test_acc_list,test_loss_list,test_fpr_lsit,test_fnr_list,test_precision_list,test_recall_list]
    name_l =["accuracy","loss","FPR","FNR","precision","recall"]


    #####################################################################################
    ####train graphs 
    count=0

    for item in train_l:
        #print(item)
        #print(epochs)
        #accuracy
        name = "train_" + name_l[count]+ ".png"
        f_path = os.path.join(path ,name)
        plt.figure(epoch+100+count)
        plt.plot(epochs,item,marker='.',label=name_l[count]+" "+vers)
        plt.legend()
        plt.savefig(f_path, bbox_inches='tight')
        count+=1

    ####test graphs
    count=0
    for item in test_l:
        #accuracy
        name = "test" + name_l[count]+ ".png"
        f_path = os.path.join(path ,name)
        plt.figure(epoch+200+count)
        plt.plot(epochs,item,label=name_l[count]+" "+vers ,marker='.')
        plt.legend()
        plt.savefig(f_path, bbox_inches='tight')
        count+=1



    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_l,test_l



######################################################################
#transfer learning model this is the model we will use for transfer learning we are retraining the last two layers of the model 


######setting up the transfer learning model 
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

layers = 18 #18 layer exei to resnet18
layercount = 0
for param in model_conv.parameters():
    if layercount == layers - 3 :
        break
    param.requires_grad = False
    layercount+=1

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


###################################################################### 

###########results of ransfer learning model 
res_trans = train_model("transfer_learning" , model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
model_conv=res_trans[0]

#storing the network 

save_dir =  'deeep_learning/IDnetwork/model_weights.pth'

torch.save(model_conv.state_dict(),save_dir)

######setting up our custom Cnn model 

custom_net = Net()


custom_net = custom_net.to(device)

criterion2 = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.

optimizer_conv2 = optim.SGD(custom_net.parameters(), lr=0.001, momentum=0.9)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer_conv2, step_size=7, gamma=0.1)


###################################################################### 

###########results of custom model 


res_custom = train_model("custom_cnn",custom_net, criterion2, optimizer_conv2,
                         exp_lr_scheduler2, num_epochs=25)
custom_net = res_custom[0]

#storing the network 

save_dir2 =  'deeep_learning/IDnetwork/model_weights_custom.pth'

torch.save(model_conv.state_dict(),save_dir2)



plt.ioff()
plt.show()
