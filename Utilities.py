import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(vals, length = 5):
    vals = np.array(vals)
    return np.convolve(vals, np.array([1/length] * length), mode = 'valid')

def plot3(losses, val_losses, val_accs):
    fix, ax = plt.subplots(3,1, figsize = (9,12))
    
    ax[0].set_title('Train Losses')
    ax[0].plot(smooth(losses))
    
    ax[1].set_title('Validation Losses')
    ax[1].plot(val_losses)
    
    ax[2].set_title('Validation Accuracies')
    ax[2].plot(val_accs)
    
    plt.show()
    
def runNTimes(N, loop):
    runs = []
    for test in range(N):
        print(f"Beginning Run {test}/{N}")
        
        losses, val_losses, val_accs = loop()           
        result = {'losses'     : losses,
                  'val_losses' : val_losses,
                  'val_accs'   : val_accs,
                  'final_perf' : val_accs[-1]}
        
        runs.append(result)
        
    return runs

def displayResults(results):
    for i, res in enumerate(results):
        print(f'Run #{i}')
        for k,v in res.items():
            print(k, ':', v)           
            
def visualizeFirstFromBatch(imgs):
    img = imgs[0].detach().cpu()
    plt.imshow(img.permute(1,2,0))
    plt.show()

def visualizeOne(dataloader):
    for x,y in dataloader:
        print('Class:', y[0].item())
        visualizeFirstFromBatch(x)
        break       
        
# Train Utilities
def trainOneEpoch(model, optimizer, criterion, train_loader, scheduler = None, epoch = 0, use_tqdm = False):
    losses = []
    train_dl = train_loader
    if use_tqdm: train_dl = tqdm(train_loader, desc = "Epoch: 0, Loss: 0")
    for x,y in train_dl:
        x,y = x.cuda(), y.cuda()

        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()

        losses.append(loss.item())
        if use_tqdm: train_dl.set_description(f'Epoch: {epoch}, Loss: {loss.item():.3f}')
    return losses


def trainOneMixupEpoch(model, optimizer, criterion, mixup_criterion, train_loader, scheduler = None, epoch = 0, use_tqdm = False):
    losses = []
    train_dl = train_loader
    if use_tqdm: train_dl = tqdm(train_loader, desc = "Epoch: 0, Loss: 0")
    for x,y in train_dl:
        x,y = x.cuda(), y.cuda()
        middle = int(np.floor(x.shape[0]/2))
        x1 = x[:middle,:,:,:]
        x2 = x[middle:2*middle,:,:,:]
        
        # Compute loss twice to compensate for the fact we are effectively halving the batch size
        optimizer.zero_grad() 
        epsilon = 1 - (np.random.power(1)) * 0.5
        x_inp = epsilon * x1 + (1-epsilon) * x2        
        predictions = model(x_inp)
        loss = mixup_criterion(predictions, y[:middle], y[middle:2*middle], epsilon)
        loss.backward()
        
        permutation = np.random.permutation(x2.shape[0])
        epsilon = 1- (1 - (np.random.power(1)) * 0.5)
        x_inp = epsilon * x1 + (1-epsilon) * x2[permutation,:,:,:]        
        predictions = model(x_inp)
        loss = mixup_criterion(predictions, y[:middle], y[middle:2*middle][permutation], epsilon)
        loss.backward()
        
        optimizer.step()
        if scheduler is not None: scheduler.step()

        losses.append(loss.item())
        if use_tqdm: train_dl.set_description(f'Epoch: {epoch}, Loss: {loss.item():.3f}') # + reg_loss.item()
    return losses

def trainOneFlatEpoch(model, optimizer, criterion, train_loader, scheduler = None, epoch = 0, use_tqdm = False):
    losses = []
    train_dl = train_loader
    if use_tqdm: train_dl = tqdm(train_loader, desc = "Epoch: 0, Loss: 0")
    for x,y in train_dl:
        x,y = x.cuda(), y.cuda()
        x = x.reshape(x.shape[0], -1)

        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()

        losses.append(loss.item())
        if use_tqdm: train_dl.set_description(f'Epoch: {epoch}, Loss: {loss.item():.3f}')
    return losses


def evalModel(model, criterion, test_loader, epoch = 0):
    losses = []
    accs = []
    
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.cuda(), y.cuda()
            
            predictions = model(x)
            loss = criterion(predictions, y)
            rounded_predictions = torch.argmax(predictions, dim = 1)
            acc = torch.sum(rounded_predictions == y).type(torch.float64) / len(rounded_predictions)
            
            losses.append(loss.item())
            accs.append(acc.item())
            
    print(f'Epoch {epoch} | Validation Loss: {np.mean(losses)} | Validation Accuracy: {np.mean(accs)}')
    
    return np.mean(losses), np.mean(accs)

def evalFlatModel(model, criterion, test_loader, epoch = 0):
    losses = []
    accs = []    
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.cuda(), y.cuda()
            x = x.reshape(x.shape[0], -1)
            
            predictions = model(x)
            loss = criterion(predictions, y)
            rounded_predictions = torch.argmax(predictions, dim = 1)
            acc = torch.sum(rounded_predictions == y).type(torch.float64) / len(rounded_predictions)
            
            losses.append(loss.item())
            accs.append(acc.item())
            
    print(f'Epoch {epoch} | Validation Loss: {np.mean(losses)} | Validation Accuracy: {np.mean(accs)}')
    
    return np.mean(losses), np.mean(accs)
