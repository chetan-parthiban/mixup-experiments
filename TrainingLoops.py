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

from Utilities import *


class ModelManager():
    
    def __init__(self, train_dir = 'data/imagenette2-320/train', val_dir = 'data/imagenette2-320/val', n_classes = 10, epochs = 5, img_size = 224):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.n_classes = n_classes
        self.epochs = epochs
        self.img_size = img_size
        
        IMG_SIZE = self.img_size
        n_classes = self.n_classes

        train_transforms = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
        train_dataset = ImageFolder(self.train_dir, transform = transforms.Compose(train_transforms))
        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 8)
        self.unaugmented_traindl = train_loader
        
        train_transforms = [transforms.RandomRotation(25), transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        train_dataset = ImageFolder(self.train_dir, transform = transforms.Compose(train_transforms))
        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 8)
        self.augmented_traindl = train_loader
        
        test_transforms = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
        test_dataset = ImageFolder(self.val_dir, transform = transforms.Compose(test_transforms))
        test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = True, num_workers = 8)
        self.test_dl = test_loader
        
        
    def train_deep_unaugmented(self):
        EPOCHS = self.epochs
        model = torchvision.models.resnet18(num_classes = self.n_classes).cuda()
        optimizer = optim.Adam(model.parameters(), lr = 5e-4, weight_decay = 0)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs = EPOCHS, steps_per_epoch = len(self.unaugmented_traindl))

        losses = []
        val_losses = []
        val_accs = []
        for epoch in range(EPOCHS):
            losses = losses + trainOneEpoch(model, optimizer, criterion, self.unaugmented_traindl, scheduler = scheduler, epoch = epoch) 
            loss, acc = evalModel(model, criterion, self.test_dl, epoch = epoch)
            val_losses.append(loss)
            val_accs.append(acc)
            
        return losses, val_losses, val_accs
    
    def train_deep_augmented(self):    
        EPOCHS = self.epochs
        model = torchvision.models.resnet18(num_classes = self.n_classes).cuda()
        optimizer = optim.Adam(model.parameters(), lr = 5e-4, weight_decay = 0)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs = EPOCHS, steps_per_epoch = len(self.augmented_traindl))

        losses = []
        val_losses = []
        val_accs = []
        for epoch in range(EPOCHS):
            losses = losses + trainOneEpoch(model, optimizer, criterion, self.augmented_traindl, scheduler = scheduler, epoch = epoch) 
            loss, acc = evalModel(model, criterion, self.test_dl, epoch = epoch)
            val_losses.append(loss)
            val_accs.append(acc)
            
        return losses, val_losses, val_accs
    
    def train_mixup(self):        
        EPOCHS = self.epochs
        model = torchvision.models.resnet18(num_classes = self.n_classes).cuda()
        optimizer = optim.Adam(model.parameters(), lr = 5e-4, weight_decay = 0)
        criterion = nn.CrossEntropyLoss()
        mixup_criterion = lambda x, y1, y2, epsilon: epsilon * criterion(x, y1) + (1-epsilon) * criterion(x, y2)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs = EPOCHS, steps_per_epoch = len(self.unaugmented_traindl))

        losses = []
        val_losses = []
        val_accs = []
        for epoch in range(EPOCHS):
            losses = losses + trainOneMixupEpoch(model, optimizer, criterion, mixup_criterion, self.unaugmented_traindl, scheduler = scheduler, epoch = epoch)
            loss, acc = evalModel(model, criterion, self.test_dl, epoch = epoch)
            val_losses.append(loss)
            val_accs.append(acc)
            
        return losses, val_losses, val_accs
    
    def train_linear(self):
        model = nn.Linear(self.image_size ** 2 * 3, self.n_classes).cuda()
        optimizer = optim.Adam(model.parameters(), lr = 1e-4)
        criterion = nn.CrossEntropyLoss()

        EPOCHS = self.epochs
        losses = []
        val_losses = []
        val_accs = []
        for epoch in range(EPOCHS):
            losses = losses + trainOneFlatEpoch(model, optimizer, criterion, self.unaugmented_traindl, epoch = epoch)
            loss, acc = evalFlatModel(model, criterion, self.test_dl, epoch = epoch)
            val_losses.append(loss)
            val_accs.append(acc)
            
        return losses, val_losses, val_accs
