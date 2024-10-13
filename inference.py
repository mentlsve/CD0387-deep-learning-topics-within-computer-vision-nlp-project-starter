# Inspired by https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import requests

# Same model as used during training with weights from fine-tuning
def model_fn(model_dir):

    # get model with weights
    model = models.resnet50(pretrained=True)

    # freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False 

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133)
    )
    
    with open(os.path.join(model_dir, "resnet50-dogs.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    model.eval()
    return model

def input_fn(request_body, content_type):

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(), # ToTensor scales to range [0.0, 1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if content_type == 'image/jpeg':
        raw_image = Image.open(BytesIO(request_body))
        transformed_image_as_tensor = eval_transform(raw_image)
        return transformed_image_as_tensor

    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Same transformations as used durint evaluation
def predict_fn(input_object, model):

    model.eval()
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0)) # this adds the batch dimension
    return prediction
