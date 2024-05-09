import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from tqdm import tqdm
import os
import random

import numpy as np
import pytorch_lightning as pl

import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    # 'IMG_SIZE': 224,
    'EPOCHS': 50,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 32*4, # 93
    'SEED': 42,
}

# Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

class CNNSuperResolutionNet(nn.Module):
    def __init__(self):
        super(CNNSuperResolutionNet, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsample1 = nn.PixelShuffle(2)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsample2 = nn.PixelShuffle(2)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Upsampling
        x = self.upsample1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        # Upsampling
        x = self.upsample2(x)
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        
        return x

class SuperResolutionLitModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = CNNSuperResolutionNet()
        self.learning_rate = CFG['LEARNING_RATE']
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, _ = batch
        img_lr = F.interpolate(img, scale_factor=0.25, mode='bicubic')
        img_sr = self.model(img_lr)
        loss = F.mse_loss(img_sr, img)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        img_lr = F.interpolate(img, scale_factor=0.25, mode='bicubic')
        img_sr = self.model(img_lr)
        loss = F.mse_loss(img_sr, img)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def prepare_data(self):
        self.imagenet_train_dataset = ImageFolder(root='/home/st/common_dataset/common_dataset/miniImageNet/Ravi/train', transform=self.transform)
        self.imagenet_val_dataset = ImageFolder(root='/home/st/common_dataset/common_dataset/miniImageNet/Ravi/val', transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.imagenet_train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.imagenet_val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)



from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Initialize TensorBoard logger
experiment_name =  'sr_cnn_mini_'+'_'.join([f'{k}_{v}' for k, v in CFG.items()]) #+ '_upscaled'
logger = TensorBoardLogger('lightning_logs', name=experiment_name)

model = SuperResolutionLitModule()
# Train the model with PyTorch Lightning
trainer = Trainer(max_epochs=CFG['EPOCHS'], logger=logger, log_every_n_steps=1, devices=[0], accelerator="gpu")
trainer.fit(model)#model, train_loader, val_loader)

