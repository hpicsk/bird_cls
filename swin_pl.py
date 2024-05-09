import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.optim.lr_scheduler as lr_scheduler
import torchvision


class SwinIRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SwinIRDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.downsample(x)

class SwinIRUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        return self.upsample(x)

class SwinIR(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        # Define the SwinIR architecture
        self.conv1 = SwinIRBlock(3, 64)
        self.downsample1 = SwinIRDownsample(64, 128)
        self.conv2 = SwinIRBlock(128, 128)
        self.downsample2 = SwinIRDownsample(128, 256)
        self.conv3 = SwinIRBlock(256, 256)
        self.upsample1 = SwinIRUpsample(256, 128, output_padding=1)
        self.conv4 = SwinIRBlock(256, 128)
        self.upsample2 = SwinIRUpsample(128, 64, output_padding=1)
        self.conv5 = SwinIRBlock(128, 64)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsample1(x)
        x = self.conv2(x)
        x = self.downsample2(x)
        x = self.conv3(x)
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.upsample2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    def training_step(self, batch, batch_idx):
        images, _ = batch
        upscaled_images = self(images)
        loss = nn.MSELoss()(upscaled_images, images)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',  # metric to monitor
            }
        }

# Load the ImageNet dataset
transform = Compose([
    Resize((256, 256)),
    # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

BATCH_SIZE=32

train_dst=torchvision.datasets.ImageFolder(root='/home/st/common_dataset/common_dataset/miniImageNet/Ravi/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dst = torchvision.datasets.ImageFolder(root='/home/st/common_dataset/common_dataset/miniImageNet/Ravi/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dst, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize the model
model = SwinIR()

# Initialize TensorBoard logger
logger = TensorBoardLogger("lightning_logs", name="swinir")

# Create the trainer
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=[0],
)

# Train the model
trainer.fit(model, train_loader, val_loader)