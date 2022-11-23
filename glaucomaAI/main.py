from utils import plot, read_ims
from model-basics import train_model
import numpy as np
import csv
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms


def glaucomaAI(data_path, csv_path, verbose=False):
    # Core function of GlaucomaAI
    # Returns the fully trained CNN
    
    # Set the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load in the data
    # Note: there should be enough memory to store all images to a local variable
    imsz = 227
    X = read_ims(data_path, imsz)
    X = X / 255
    Y = np.zeros((X.shape[0],4))
    
    # Load in the labels
    index = 0
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                if verbose: print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                line_count += 1
                if(index < Y.shape[0]):
                    Y[index, int(row[1])] = 1
                index+=1
        if verbose: print(f'Processed {line_count} lines.')
        
    
    # Random shuffle of data and labels
    r = np.random.permutation(X.shape[0])
    X = X[r,:,:,:]
    Y = Y[r,:]
    
    X_train = X[0:int(0.8*X.shape[0])]
    X_val = X[int(0.8*X.shape[0]):X.shape[0]] 
    
    Y_train = Y[0:int(0.8*Y.shape[0])]
    Y_val = Y[int(0.8*Y.shape[0]):Y.shape[0]] 
    
    
    # Data transforms
    Data = {'train':X_train,'val':X_val}
    Labels = {'train':Y_train,'val':Y_val}
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation([-15, 15], resample=False, expand=False, center=None),
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
    
    
    # Define data loader
    image_datasets = {x: torch.utils.data.TensorDataset(torch.tensor(Data[x], dtype=torch.float).transpose(3,1),torch.tensor(Labels[x], dtype=torch.float)) for x in ['train', 'val']}
    dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes  = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    
    # Initialize neural network
    net = resnext101_32x8d(pretrained=True).to(device)
    
    # Define optimizer and learning rate scheduler
    optimizer_ft = optim.SGD(net.parameters(), lr=0.003162, momentum=0.75)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.9)
    
    # Train the neural network!
    net = train_model(net, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)
    
    return net
    
    