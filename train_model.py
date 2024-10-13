#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

import argparse
import os

import smdebug.pytorch as smd

def test(model, test_loader, criterion, hook):
    
    test_loss_accumulator = 0
    test_correct_accumulator= 0
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    
    for batch_data, batch_target in test_loader:
        batch_pred = model(batch_data) # forward pass
        batch_loss = criterion(batch_pred, batch_target) # calculate loss
        test_loss_accumulator += batch_loss
        batch_pred = batch_pred.argmax(dim=1, keepdim=True) # pred is a 2-d tensor of shape (batch_size, 1) which contains the number of the predicted class
        test_correct_accumulator += batch_pred.eq(batch_target.view_as(batch_pred)).sum().item() # add 1 for every correct predicted image
    
    print(f"Test: Accuracy: {100* (test_correct_accumulator/len(test_loader.dataset))}%")
    print(f"Test: Average loss: {(test_loss_accumulator/len(test_loader.dataset))}")
    

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, hook):
    
    for epoch in range(epochs):
        
        # Training
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad() # reset gradients
            batch_pred = model(batch_data) # forward pass
            batch_loss = criterion(batch_pred, batch_target) # calculate loss
            batch_loss.backward() # calculate gradients
            optimizer.step() # update weights

        # Validation
        epoch_validation_loss_accumulator = 0
        epoch_validation_correct_accumulator = 0
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        
        with torch.no_grad():
            for batch_data, batch_target in validation_loader:
                batch_pred = model(batch_data) # forward pass
                batch_loss = criterion(batch_pred, batch_target) # calculate loss
                epoch_validation_loss_accumulator += batch_loss
                batch_pred = batch_pred.argmax(dim=1, keepdim=True) # pred is a 2-d tensor of shape (batch_size, 1) which contains the number of the predicted class
                epoch_validation_correct_accumulator += batch_pred.eq(batch_target.view_as(batch_pred)).sum().item() # add 1 for every correct predicted image
        
        print(f"Epoch #{epoch}: Validation accuracy: {100* (epoch_validation_correct_accumulator/len(validation_loader.dataset))}%")
        print(f"Epoch #{epoch}: Validation average loss: {(epoch_validation_loss_accumulator/len(validation_loader.dataset))}")

    return model
    
def net():
    # get model with weights
    model = models.resnet50(pretrained=True)

    # freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False 

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    return model

def is_file_an_image(path):
  try:
    im = Image.open(path)
    return True
  except:
    print(f"File: {path} is not an image. Ignoring")
    return False


def create_data_loaders(batch_size):
    
    train_data_path = os.environ['SM_CHANNEL_TRAIN']
    test_data_path = os.environ['SM_CHANNEL_TEST']
    validation_data_path = os.environ['SM_CHANNEL_VALIDATION']

    # From https://pytorch.org/vision/stable//models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    # ResNet50_Weights.IMAGENET1K_V2:
    # The images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224]. 
    # Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(), # ToTensor scales to range [0.0, 1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(), # ToTensor scales to range [0.0, 1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Every directory in the directory root is considered a class, and every file in a 'class' directory is an observation/image.
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform, is_valid_file=is_file_an_image)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=eval_transform, is_valid_file=is_file_an_image)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=eval_transform, is_valid_file=is_file_an_image)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False) 
    
    return train_data_loader, test_data_loader, validation_data_loader


def main(args):

    model=net()

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    train_loader, test_loader, valid_loader = create_data_loaders(args.batch_size)
    
    loss_criterion = nn.CrossEntropyLoss()
    hook.register_loss(loss_criterion)
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    print("Start model training")
    model = train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, hook)

    print("Start model testing")
    test(model, test_loader, loss_criterion, hook)
    
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'resnet50-dogs.pth'))

if __name__=='__main__':

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()

    print(f"Entry point parsed arguments {args}")
    print(f"Environment variable {os.environ['SM_CHANNEL_TRAIN']}")
    main(args)
