# first import the module
import webbrowser
import cv2
import os 

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


vid = cv2.VideoCapture(0)

#video metadata
fps = vid.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)
count = 0
while(count<10):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	path = 'deeep_learning/check_photos/person'
	
	pathtowrite = os.path.join(path, "frame%d.jpg" % count)
	count+=1
	cv2.imwrite(pathtowrite, frame)
		

		#print(os.path.join(path, 'val' , "frame%d.jpg" % count) )
		#cv2.imwrite("frame%d.jpg" % count, frame)
	# Display the resulting frame
	cv2.imshow('frame', frame)
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

####loading the transfer learning model 
PATh1 = 'deeep_learning/IDnetwork/model_weights.pth'
#####using the same model as in training
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
layers = 18 #18 layer exei to resnet18
layercount = 0
for param in model.parameters():
    if layercount == layers - 3 :
        break
    param.requires_grad = False
    layercount+=1
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

######loading the weights from training 
model.to(device)
model.load_state_dict(torch.load(PATh1))
model.eval()

#####using the same transformations as in training 
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


#this is the directory where i have saved my images 
dir  = 'deeep_learning/check_photos'
image_datasets =  datasets.ImageFolder(dir, data_transforms)

###i want a 70% succes on the images i took in order to open the web 
count=0
sum =0
for photo,label in image_datasets:
    photo2 = photo.unsqueeze(0).to(device)
    res = model(photo2)
    _, preds = torch.max(res, 1)
    print(preds)
    count+=1
    sum+=preds.item()

###putting the threshold to open the camera 
threshold = 0.7
if sum >=threshold*count:
    print("Welcome Master")
    # then make a url variable
    url = "https://store.steampowered.com/"
    
    # then call the get method to select the code 
    # for new browser and call open method 
    # described above
    webbrowser.open(url)
    
    # results in error since chrome is not registered initially.
else:
    print("You are not authorized to access the web")
	

