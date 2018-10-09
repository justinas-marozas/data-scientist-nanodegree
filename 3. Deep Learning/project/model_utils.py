import copy
import json
from time import time
from collections import OrderedDict

import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from PIL import Image

# from workspace_utils import active_session


def get_color_normalization_transform():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

def get_prediction_data_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        get_color_normalization_transform()
    ])

def build_data_loaders(data_dir, batch_size=32):
    train_dir = f'{data_dir}/train'
    validate_dir = f'{data_dir}/valid'
    test_dir = f'{data_dir}/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            get_color_normalization_transform()
        ]),
        'validate': get_prediction_data_transforms(),
        'test': get_prediction_data_transforms()
    }

    # load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'validate': datasets.ImageFolder(validate_dir, transform=data_transforms['validate']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # using the image datasets and the trainforms, define the dataloaders
    data_loaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'validate': DataLoader(image_datasets['validate'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=True)
    }
    
    return data_loaders


def get_available_architectures():
    return {
        'alexnet': models.alexnet,
        'vgg-11': models.vgg11,
        'vgg-13': models.vgg13,
        'vgg-16': models.vgg16,
        'vgg-19': models.vgg19,
        'densenet-121': models.densenet121,
        'densenet-161': models.densenet161,
        'densenet-169': models.densenet169,
        'densenet-201': models.densenet201,
        'resnet-18': models.resnet18,
        'resnet-34': models.resnet34,
        'resnet-50': models.resnet50,
        'resnet-101': models.resnet101,
        'resnet-152': models.resnet152
    }

def build_model(architecture, classifier_hidden_units, class_count):
    architecture = architecture.lower()
    try:
        constructor = get_available_architectures()[architecture]
    except KeyError:
        keys = ', '.join(get_available_architectures().keys())
        raise Error(f'Invalid model archutecture. Try one of {keys}.')
    model = constructor(pretrained=True)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    # some pretrained model classifiers have a single layer and some have multiple
    # capture the first input of the classifier
    try:
        classifier_inputs = model.classifier[0].in_features
    except TypeError:
        classifier_inputs = model.classifier.in_features
    
    classifier = build_classifier(classifier_inputs, class_count, classifier_hidden_units)
    
    model.classifier = classifier
    
    return model

def build_classifier(input_size, output_size, hidden_layers):
    dropout_rate = 0.25
    
    layers = OrderedDict()
    layers_in_out = zip([input_size] + hidden_layers, hidden_layers)

    for i, (l_in, l_out) in enumerate(layers_in_out):
        layers[f'{i}-fc'] = nn.Linear(in_features=l_in, out_features=l_out, bias=True)
        layers[f'{i}-relu'] = nn.ReLU()
        layers[f'{i}-dropout'] = nn.Dropout(p=dropout_rate)

    layers['output'] = nn.Linear(in_features=hidden_layers[-1], out_features=output_size, bias=True)
    layers['output-norm'] = nn.LogSoftmax(dim=1)

    return nn.Sequential(layers)

def save_checkpoint(checkpoint_filepath, model, model_architecture):
    try:
        classifier_input_units = model.classifier[0].in_features
        classifier_hidden_units = [_.out_features for _ in model.classifier if type(_) is nn.Linear]
        classifier_output_units = classifier_hidden_units[-1]
        classifier_hidden_units = classifier_hidden_units[:-1]
    except TypeError:
        classifier_input_units = model.classifier.in_features
        classifier_output_units = model.classifier.out_features
        classifier_hidden_units = []
    
    checkpoint = {
        'architecture': model_architecture,
        'classifier': {
            'input_units': classifier_input_units,
            'output_units': classifier_output_units,
            'hidden_units': classifier_hidden_units
        },
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, checkpoint_filepath)

def load_from_checkpoint(checkpoint_filepath):
    checkpoint = torch.load(checkpoint_filepath)
    architecture = checkpoint['architecture']
    class_count = checkpoint['classifier']['output_units']
    classifier_hidden_units = checkpoint['classifier']['hidden_units']
    
    model = build_model(architecture, classifier_hidden_units, class_count)
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.features.parameters():
        param.requires_grad = False
        
    return model

def get_processing_device(use_gpu=False):
    if use_gpu:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return 'cpu'

def run_model(model, phases, criterion, optimizer, scheduler, num_epochs, device, data_loaders, dataset_sizes):
    since = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            since_phase = time()
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_accuracy = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_accuracy += (preds == labels.data).type(torch.FloatTensor).mean()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_accuracy / dataset_sizes[phase]

            print(f'{phase} loss: {epoch_loss:.4f} accuracy: {epoch_acc:.4f} duration: {time() - since_phase:.1f}s')

            # deep copy the model
            if phase != 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time() - since
    print('Session complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy value: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train(model, num_epochs, learning_rate, use_gpu, data_loaders):
    device = get_processing_device(use_gpu)
    print(f'Running model on device: {device}.')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    dataset_sizes = {x: len(data_loaders[x]) for x in ['train', 'validate', 'test']}
    
    # with active_session():
    model = run_model(
        model,
        ['train', 'validate'],
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        device,
        data_loaders,
        dataset_sizes)

    test(
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        data_loaders,
        dataset_sizes)

    return model

def test(model, criterion, optimizer, scheduler, device, data_loaders, dataset_sizes):
    # with active_session():
    run_model(
        model,
        ['test'],
        criterion,
        optimizer,
        scheduler,
        1,
        device,
        data_loaders,
        dataset_sizes)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transforms = get_prediction_data_transforms()
    image = transforms(image)
    return image

def predict(image_path, model, use_gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image = process_image(image)
    image.unsqueeze_(0)
    
    device = get_processing_device(use_gpu)
    image = image.to(device)
    model = model.to(device)
    model.eval()
    
    output = model(image)

    probs, classes = torch.topk(output, topk)
    # get probabilities from logsoftmax-ed output
    probs = torch.exp(probs)
    # mundane python arrays are expected as an output
    probs, classes = [_ for _ in probs[0].detach().cpu().numpy()], [_ for _ in classes[0].detach().cpu().numpy()]
    return probs, classes

def get_category_labels(map_filepath):
    with open(map_filepath, 'r') as f:
        cat_lbl_map = json.load(f)
        cat_lbl_map = {int(key):val for key, val in cat_lbl_map.items()}
    return cat_lbl_map
