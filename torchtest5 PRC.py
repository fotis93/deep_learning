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

cudnn.benchmark = True
plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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

data_dir =  '/home/fotis/IDphotos'
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

#print(image_datasets)
#print(class_names)


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])



######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_fpr = 0
    best_fnr = 0
    train_acc_list= []
    test_acc_list = []

    train_loss_list = []
    test_loss_list = []

    train_fpr_list = []
    test_fpr_lsit = []

    train_fnr_list = []
    test_fnr_list = []


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



            #####calculating roc curve 

            #print(tot_labels)
            #print(tot_prob)
            #print(fotis)

            ########creating the curves for each epoch 
            '''
            fpr, tpr, thresholds = metrics.roc_curve(tot_labels, tot_prob)

            path = '/home/fotis/roc_curves'
            f_path = os.path.join(path , "epoch%d.jpg" % epoch)
            plt.figure(epoch)
            plt.plot(fpr, tpr,label=f'{phase}',marker='.')
            plt.legend()
            plt.savefig(f_path, bbox_inches='tight')
            #plt.show()


            precision, recall, thresholds = metrics.precision_recall_curve(tot_labels, tot_prob)

            path = '/home/fotis/prc_curves'
            f_path = os.path.join(path , "epoch%d.jpg" % epoch)
            plt.figure(epoch+25)
            plt.plot(recall,precision,label=f'{phase}',marker='.')
            plt.legend()
            plt.savefig(f_path, bbox_inches='tight')
            #plt.show()
            '''



                #running_ones = 
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            FPR = False_positives/(False_positives + True_negatives)
            FNR = False_negatives/(False_negatives + True_positives)

            if phase == "train":
                    train_acc_list.append(epoch_acc.item())
                    train_loss_list.append(epoch_loss)
                    train_fpr_list.append(FPR)
                    train_fnr_list.append(FNR)
            else:
                    test_acc_list.append(epoch_acc.item())
                    test_loss_list.append(epoch_loss)
                    test_fpr_lsit.append(FPR)
                    test_fnr_list.append(FNR)



            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} FPR: {FPR:.4f} FNR: {FNR:.4f}  ')


            # if i do not have better model i want to break the loop
           
            ''' if phase == 'val'and breakcount == 5:
                best_model_wts = copy.deepcopy(model.state_dict())
                return model
            if phase == 'val':
                breakcount+=1'''
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_fpr = FPR
                best_fnr = FNR
                best_model_wts = copy.deepcopy(model.state_dict())
                breakcount = 0
            '''if phase == 'val':
                print(breakcount)'''
            

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Best val FPR: {best_fpr:4f}')
    print(f'Best val FPR: {best_fnr:4f}')

    #######training graphs 
    path = '/home/fotis/graphs'
    epochs = [i for i in range(25)]

    train_l = [train_acc_list,train_loss_list,train_fpr_list,train_fnr_list]
    test_l = [test_acc_list,test_loss_list,test_fpr_lsit,test_fnr_list]
    name_l =["accuracy","loss","FPR","FNR"]

    count=0
    ####train graphs
    for item in train_l:
        #print(item)
        #print(epochs)
        #accuracy
        name = "train_" + name_l[count]+ ".png"
        f_path = os.path.join(path ,name)
        plt.figure(epoch+100+count)
        plt.plot(epochs,item,marker='.',label=name_l[count])
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
        plt.plot(epochs,item,label=name_l[count],marker='.')
        plt.legend()
        plt.savefig(f_path, bbox_inches='tight')
        count+=1



    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



######################################################################
# ConvNet as fixed feature extractor
# ----------------------------------
#
# Here, we need to freeze all the network except the final layer. We need
# to set ``requires_grad = False`` to freeze the parameters so that the
# gradients are not computed in ``backward()``.
#
# You can read more about this in the documentation
# `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
#

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')

layers = 18 #18 layer exei to resnet18
layercount = 0
for param in model_conv.parameters():
    if layercount == layers - 3 :
        break
    param.requires_grad = False
    layercount+=1

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

#storing the network 

save_dir =  '/home/fotis/IDnetwork/model_weights.pth'

torch.save(model_conv.state_dict(),save_dir)


######################################################################
#

visualize_model(model_conv)

plt.ioff()
plt.show()
