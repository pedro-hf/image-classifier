import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from PIL import Image

def get_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(45),
                                              transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_datasets, valid_datasets, test_datasets

def create_classifier(inputs, outputs, hidden_layers, p):
    layers = []
    last_layer_size = inputs
    for i, layer_size in enumerate(hidden_layers):
        layer = [(str(i) + '_layer', nn.Linear(last_layer_size, layer_size)),
                (str(i) + '_relu', nn.ReLU()),
                (str(i) + '_dropout', nn.Dropout(p))]
        layers += layer
        last_layer_size = layer_size
    output_layer = [('out_layer', nn.Linear(last_layer_size, outputs)),
                    ('output', nn.LogSoftmax(dim=1))]
    layers += output_layer
    classifier = nn.Sequential(OrderedDict(layers))
    return classifier

def initialize_model(model, output_size, hidden_layers, p, lr):
    if model == 'vgg':
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        input_size = model.classifier[0].in_features
        model.classifier = create_classifier(input_size, output_size, hidden_layers, p)
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif model == 'resnet':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        input_size = model.fc.in_features
        model.fc = create_classifier(input_size, output_size, hidden_layers, p)
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    elif model == 'densenet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        input_size = model.classifier.in_features
        model.classifier = create_classifier(input_size, output_size, hidden_layers, p)
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        print('Model %s is not available' % model)
        print('Choose between : vgg, densenet or resnet')
        return None
    return model, optimizer

def save_model(model, model_type, save_dir):
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'model_type': model_type}
    if model_type == 'vgg' or model_type == 'densenet':
        checkpoint['classifier'] = model.classifier
    elif model_type == 'resnet':
        checkpoint['classifier'] = model.fc
    else:
        print('Model not available, choose between : vgg, densenet or resnet')
        return None
    torch.save(checkpoint, save_dir)

def validation(model, test_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()   
    return test_loss, accuracy

def load_checkpoint(file):
    checkpoint = torch.load(file)
    model_type = checkpoint['model_type']
    if model_type == 'vgg':
        model = models.vgg11()
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
    elif model_type == 'densenet':
        model = models.densenet121()
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
    elif model_type == 'resnet':
        model = models.resnet18()
        model.fc = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('%s no such model' % model_type)
    return model

def process_image(image):
    # Scale to 255
    w, h = image.size
    r = 255/min(w,h)
    w = int(r * w)
    h = int(r * h)
    image = image.resize((w, h))
    # center crop to 224
    left = int((w - 224)/2)
    upper = int((h - 224)/2)
    right = int((w + 224)/2)
    lower = int((h + 224)/2)
    image = image.crop((left , upper, right, lower))

    # Transform to numpy array
    np_image = np.array(image)
    np_image = np_image/255
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))
    return np_image

def predict(image_path, model, device, topk):
    with torch.no_grad():
        image = Image.open(image_path)
        image = torch.from_numpy(process_image(image)).float()
        output = model(image.unsqueeze(0).to(device))
        probs, classes = output.topk(topk)
        ps = np.exp(probs.cpu().data.numpy()[0])
        idx_to_class = {i: c for c, i in model.class_to_idx.items()}
        clas = np.array([str(idx_to_class[x]) for x in classes.cpu().data.numpy()[0]])
        return ps, clas