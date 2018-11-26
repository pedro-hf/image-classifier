# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

The application is divided into 3 python files:
* train: called on a directory it will train a model on the data stored in that directory. Optional arguments:
	* --arch: choose between 'vgg' for vgg11, 'resnet' for resnet18 or 'densenet' for 121.
    * --hiden_units: list of hidden unit layers on the model's classifier
    * --learning_rate: learning rate during training.
    * --drop_out: probability of droput for each layer
    * --gpu: runs on the GPU instead of the CPU
    * --epochs: number of epochs in the training
    * --save_dir: directory where to save the checkpoint.pth file.
* predict: Used to classify an image with the model saved in checkpoint. Optional arguments:
	* --top_k: integer, prints out the top k probabilities given by the model.
    * --category_names: .json file that maps class prediction to a name.
    * --gpu: run on the GPU instead of the CPU
* help_functions: help functions used in the other two files.