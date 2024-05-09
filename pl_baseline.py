# %% 
import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

# Hyperparameter Setting
CFG = {
    # 'IMG_SIZE': 224,
    'EPOCHS': 50,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 32, # 93
    'SEED': 42,
    'SPLIT_PROPORTION': 0.2,
    'BASE_ARCH':'swin_v2_s' # efficientnet_v2_l efficientnet_b7 resnet50 swin_v2_s 
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

# Train & Validation Split
df = pd.read_csv('./train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=CFG['SPLIT_PROPORTION'], stratify=df['label'], random_state=CFG['SEED'])

# Label-Encoding
le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])

# CustomDataset
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None, preprocess=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.preprocess = preprocess

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        # import pdb; pdb.set_trace()
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.preprocess is not None:
            
            image = self.preprocess(image)

        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)

# torchvision.models transform
if CFG['BASE_ARCH'] == 'resnet50':
    weights = models.ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
elif CFG['BASE_ARCH'] == 'efficientnet_v2_l':
    preprocess = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
elif CFG['BASE_ARCH'] == 'efficientnet_b7':
    preprocess = models.EfficientNet_B7_Weights.IMAGENET1K_V1.transforms()
elif CFG['BASE_ARCH'] == 'swin_v2_s':
    preprocess = models.Swin_V2_S_Weights.IMAGENET1K_V1.transforms()

# Transformations
train_transform = A.Compose([
    # A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2(),
    ])

test_transform = A.Compose([
    # A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2(),
    ])

train_dataset = CustomDataset(train['upscale_img_path'].values, train['label'].values, train_transform, preprocess=preprocess)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val['upscale_img_path'].values, val['label'].values, test_transform, preprocess=preprocess)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# Test loader
# test = pd.read_csv('./test.csv')
test = pd.read_csv('./test_upscale.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform, preprocess=preprocess)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


# Model Definition
class BaseModel(pl.LightningModule):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = getattr(models, CFG['BASE_ARCH'])(weights='DEFAULT') # models.mobilenet_v3_small(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = CFG['LEARNING_RATE']

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self(imgs)
        loss = self.criterion(output, labels) #* 100 # for amplifyingloss
        loss = loss*10
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self(imgs)
        loss = self.criterion(output, labels)
        loss = loss*10
        preds = output.argmax(1)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'preds': preds, 'labels': labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([output['preds'] for output in outputs])
        labels = torch.cat([output['labels'] for output in outputs])
        val_score = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        self.log('val_f1_score', val_score, prog_bar=True, logger=True)

        # # Compute confusion matrix
        # conf_matrix = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())

        # # Plot confusion matrix
        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
        # ax.set_title('Confusion Matrix')
        # ax.set_xlabel('Predicted Labels')
        # ax.set_ylabel('True Labels')
        # self.logger.experiment.add_figure('Confusion Matrix', fig, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_f1_score'}

# Instantiate the model
model = BaseModel()

# Initialize TensorBoard logger
experiment_name =  '_'.join([f'{k}_{v}' for k, v in CFG.items()]) + '_upscaled'
logger = TensorBoardLogger('lightning_logs', name=experiment_name)

# Train the model with PyTorch Lightning
trainer = pl.Trainer(max_epochs=CFG['EPOCHS'], logger=logger, log_every_n_steps=1, devices=[0], accelerator="gpu")
# trainer.fit(model, train_loader, val_loader)

# # load ckpt
# ckpt_path = '/home/st/dacon/bird_cls/lightning_logs/baseline_model/version_3/checkpoints/epoch=49-step=17350.ckpt'
# model = BaseModel.load_from_checkpoint(ckpt_path)
# model.eval()

preds = trainer.predict(model, test_loader)
preds = torch.cat([output for output in preds])
# import pdb; pdb.set_trace()
preds = le.inverse_transform(preds.argmax(1).cpu().numpy())  # Reshape preds to be 1-dimensional

# Submission
submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds
submit.to_csv(f'./ups_submit_{experiment_name}.csv', index=False)

