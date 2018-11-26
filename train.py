import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from help_functions import get_loaders, initialize_model, validation, save_model
from workspace_utils import active_session

# Create parser for input arguments 
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('data_directory', action='store', type=str)
parser.add_argument('--save_dir', action='store', default='checkpoint.pth', type=str)
parser.add_argument('--learning_rate', action='store', default=0.001, type=float)
parser.add_argument('--drop_out', action='store', default=0.5, type=float)
parser.add_argument('--arch', action='store', default='vgg', type=str)
parser.add_argument('--hidden_units', action='store', nargs='+', default=[1000,500], type=int)
parser.add_argument('--epochs', action='store', default=3, type=int)
parser.add_argument('--gpu', action='store_true')
inputs = parser.parse_args()

#get data loaders from the abligatory input 'data_directory'
data_dir = inputs.data_directory
train_datasets, valid_datasets, test_datasets = get_loaders(data_dir)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
#create the pretrained model, change the classifier and define an optimizer. add dictionary that relates class to index.
number_of_outputs = len(train_datasets.class_to_idx)
model, optimizer = initialize_model(inputs.arch, number_of_outputs, inputs.hidden_units,
                         inputs.drop_out, inputs.learning_rate)
model.class_to_idx = train_datasets.class_to_idx
# Define the loss
criterion = nn.NLLLoss()

#Train model, looping through the training images
print_every = 5
steps = 0
if inputs.gpu:
    device = 'cuda'
else:
    device ='cpu'
model.to(device)
model.train()
with active_session():
    for e in range(inputs.epochs):
        running_loss = 0
        for ii, (images, labels) in enumerate(train_loader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, test_loader, criterion, device)
                model.train()
                print("Epoch: {}/{}... ".format(e+1, inputs.epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every), 
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
                running_loss = 0

# Save the model
save_model(model, inputs.arch, inputs.save_dir)
