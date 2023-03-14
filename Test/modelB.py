
import torch
from torchvision.transforms import functional as F

import os
import csv
import sys
import pandas as pd
from PIL import Image
import numpy as np
import cv2

#from torchsummary import summary
#from matplotlib import pyplot as plt

current_directory = os.getcwd()
home_directory = os.path.dirname(current_directory)


#  insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, os.path.join(home_directory,"Scripts","Necessary_files"))

#import transforms as T
import torchvision.transforms as transforms
import utils


log_csv_file_directory = os.path.join(home_directory, "Log")
log_csv_fieldName = ["Epoch", "Learning Rate", "Total training loss", "Total validation loss"]
 
BATCH = 128
final_epoch = 10000

# DO you want to continute training by loading a model?
continue_training = False

# If contineous training is True then below make sense
epoch_to_load = 0
checkpoint_name = "trained_epoch{}.pth".format(epoch_to_load)
checkpoint_path = os.path.join(log_csv_file_directory,"saved_callback_each_epoch")

data_filename = "data.csv"
data_location_and_filename = os.path.join(home_directory,"Data",data_filename)
data = pd.read_csv(data_location_and_filename)

def mask_file_names(data):
    imgs_names = list( sorted(data["External ID"]) )
    mask_names = []
    for idx in range(len(imgs_names)):
        mask_names.append(os.path.splitext(imgs_names[idx])[0] + "_mask.jpg")
    return mask_names


transform_img = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.Normalize((0.56, 0.46, 0.44),(0.22, 0.198, 0.204))
    ])

transform_mask = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.Normalize((0.235,),(0.423,))
    ])


class SegmetationDataset(object):
    
    #Data is a csv file with all required informations
    def __init__(self, img_transform=None, mask_transform=None):
    #def __init__(self):

        #List name of all images
        self.imgs_names = list( sorted(data["External ID"]) ) 
        self.mask_names = mask_file_names(data)   
        self.img_transform  = img_transform
        self.mask_transform = mask_transform

    
    def __getitem__(self, idx):
        
        img_path   = os.path.join(home_directory, "Dataset", "Reshaped", "Images", self.imgs_names[idx])
        mask_path  = os.path.join(home_directory, "Dataset", "Reshaped", "Masks",  self.mask_names[idx])

        img  = Image.open(img_path)
        mask = Image.open(mask_path).convert('L')

        if self.img_transform is not None:
            img  = self.img_transform(img)
            
        if self.mask_transform is not None:  
            mask = self.mask_transform(mask)

        #img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)/255
        #mask = cv2.cvtColor(cv2.imread(mask_path),cv2.COLOR_BGR2GRAY)

        #ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        #mask = mask/255
        
        #img  = torch.from_numpy(img).float().permute(2,0,1)
        #mask = torch.unsqueeze(torch.from_numpy(mask).float(),0)
        
        # Convert to tensor and return
        return img, mask
    
    def __len__(self):
        return len(self.imgs_names)



def validation_func(model, validateloader, device, loss_function):

    val_loss = 0.0

    for test_imgs,test_targets in validateloader:

        test_images  = torch.stack(test_imgs)
        test_targets = torch.stack(test_targets)

        test_images  = test_images.to(device)
        test_targets = test_targets.to(device)

        total_loss = loss_function(model.forward(test_images), test_targets)
        val_loss = val_loss + total_loss

    return val_loss.detach().numpy() 


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def unet_left_convolutions(inputChannels, outputChannels):
    
    conv = torch.nn.Sequential(
        
        torch.nn.Conv2d(inputChannels, outputChannels, kernel_size=5),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.Conv2d(outputChannels, outputChannels, kernel_size=1),
        torch.nn.BatchNorm2d(outputChannels),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    )
    
    return conv

def unet_right_convolutions(inputChannels, outputChannels):
    
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(inputChannels, int(outputChannels*2), kernel_size=1, padding=1),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.ConvTranspose2d(int(outputChannels*2), outputChannels, kernel_size=2, stride=2),
        torch.nn.BatchNorm2d(outputChannels),
        torch.nn.ReLU(inplace=True),
    )
    
    return conv


def final_convolutions(inputChannels, outputChannels):
    
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(inputChannels, outputChannels,  kernel_size=1, padding=0),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.Conv2d(outputChannels, outputChannels, kernel_size=1, padding=0),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.Conv2d(outputChannels, outputChannels, kernel_size=1, padding=0),
        torch.nn.ReLU(inplace=True),
    )
    return conv

def down_convolutions(inputChannels, outputChannels):
    
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(inputChannels, outputChannels,  kernel_size=3, padding=0),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.Conv2d(outputChannels, outputChannels, kernel_size=3, padding=0),
        torch.nn.ReLU(inplace=True),
        
        torch.nn.Conv2d(outputChannels, 1, kernel_size=1, padding=0),
        #torch.nn.ReLU(inplace=True),
    )
    return conv



class Model(torch.nn.Module):
    
    def __init__(self):
        
        super(Model, self).__init__()
        
        self.layer1L = unet_left_convolutions(3,8)
        self.layer2L = unet_left_convolutions(8,16)
        self.layer3L = unet_left_convolutions(16,32)
        self.layer4L = unet_left_convolutions(32,64)
        self.layer5L = unet_left_convolutions(64,128)
        
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        self.layer6L = torch.nn.Conv2d(128, 256, (1,1))
        self.layer6R = torch.nn.Conv2d(256, 128, (1,1))
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        
        self.concatenate = lambda left,right: torch.cat((left,right),1)
        
        self.layer5R = unet_right_convolutions(256, 64)
        self.layer4R = unet_right_convolutions(128, 32)
        self.layer3R = unet_right_convolutions(64, 16)
        self.layer2R = unet_right_convolutions(32, 8)
        self.layer1R = unet_right_convolutions(16, 4)
        
        #self.layer1post = final_convolutions(4, 4)
        #self.output  = torch.nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)

        self.layer1post = down_convolutions(4,4)

    def forward(self, x):
        
        x1L = self.layer1L(x)
        x2L = self.layer2L(x1L)
        x3L = self.layer3L(x2L)
        x4L = self.layer4L(x3L)
        x5L = self.layer5L(x4L)
        
        # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        x6L  = self.layer6L(x5L)
        x6R  = self.layer6R(x6L)
        # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        
        #print("x5 concatenate", self.concatenate(x5L, x6R).shape )
        x5R = self.layer5R( self.concatenate(x5L, x6R) )
        #print("x4 concatenate", self.concatenate(x4L, x5R).shape )
        x4R = self.layer4R( self.concatenate(x4L, x5R) )
        #print("x3 concatenate", self.concatenate(x3L, x4R).shape )
        x3R = self.layer3R( self.concatenate(x3L, x4R) )
        #print("x2 concatenate", self.concatenate(x2L, x3R).shape )
        x2R = self.layer2R( self.concatenate(x2L, x3R) )
        #print("x2 concatenate", self.concatenate(x1L, x2R).shape )
        x1R = self.layer1R( self.concatenate(x1L, x2R) )
        
        #last_1by1 = self.layer1post(x1R)
        #result    = self.output(last_1by1)  #1*1 convultion 572*572*1

        result = self.layer1post(x1R)
        return result

# --------- Note sigmoid is embedded in the loss function

def main():

    # ---------------> Dataloader <--------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataSet = SegmetationDataset(transform_img,transform_mask)
    #dataSet = SegmetationDataset()

    torch.manual_seed(0)
    train_size = int(0.8 * len(dataSet))
    test_size = len(dataSet) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataSet, [train_size, test_size])

    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True,
                                            collate_fn=utils.collate_fn)

    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, shuffle=True,
                                            collate_fn=utils.collate_fn)
    
    
    # ---------------> Retraining the MODEL <--------------------------
    print("continue_training",continue_training)
    if (continue_training == True):

        # In the case of continuing the training: Load the saved file
        loaded_checkpoint = torch.load(os.path.join(checkpoint_path,checkpoint_name))
        model.load_state_dict(loaded_checkpoint["model_state"])
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
        current_epoch = loaded_checkpoint["epoch"]

    else:

        current_epoch = 0


    model = Model()    

    #Move the model to the device
    model.to(device)    
    
    
    # ----> Create an Empty CSV File with headers which stores training and validation <----------- 
    log_file_name = "training_data_log_starting_epoch{}.csv".format(current_epoch) #This is the name of the csv file
    with open(os.path.join(log_csv_file_directory,log_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(log_csv_fieldName)

    file.close()
    
    
    
    # ------------>>>>>>>> OPTIMIZATION <<<<<<---------------
    
    #1. Get parameters

    params = [p for p in model.parameters() if p.requires_grad]
    #print(params)

    #2. Define Optimization algorithm
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay= 0.0001)

    #Learning rate sceduling
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9,
    #                                                  patience=10,
    #                                                  verbose=False)

    loss_function = torch.nn.BCEWithLogitsLoss()
    
    # ---------------> TRAINING MODEL <--------------------------
    
    for epoch in np.arange(current_epoch, final_epoch+1, 1):

        # Keep in model in training mode:
        model.train()

        training_loss = 0.0

        # This gives a minibatch at a time
        #for images, targets in data_loader:

        for images, targets in train_loader:

            images  = torch.stack(images)
            targets = torch.stack(targets)

            #Since our model can be in either CPU or GPU we transfer to the respective branch
            images = images.to(device)
            targets= targets.to(device)

            total_loss = loss_function(model(images), targets)

            # 3. Run the optimizer and and compute the loss   --- for this minibatch
            optimizer.zero_grad()
            total_loss.backward()

            # 4. This does the update of parameters --- for this minibatch
            optimizer.step()

            #print("I finished one minibatch")

            # Sum all the training loss value
            training_loss += total_loss.item()

        
        model.eval()
        with torch.no_grad():
            validation_loss = validation_func(model, test_loader, device, loss_function)

        with open(os.path.join(log_csv_file_directory,log_file_name), "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_csv_fieldName)
            writer.writerow({"Epoch": epoch, "Learning Rate":get_lr(optimizer),"Total training loss": training_loss/len(train_loader),
                            "Total validation loss": validation_loss/len(test_loader)})


        #lr_scheduler.step(validation_loss)

        # ------------------------ SAVING ---------------------- #

        if (epoch % 250) == 0:

            #We bring both model back to cpu and save it
            #Save each callback
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }


            #Save the parameters values needed
            torch.save(checkpoint, os.path.join(log_csv_file_directory,"saved_callback_each_epoch",
                                                "trained_epoch{}.pth".format(epoch)))
    

main()


