# COMPONENT IMPORTS (Data handling, model definition)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
from google.colab import drive

# TRAINING IMPORTS (Optimization, scheduling, monitoring)
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

# EVALUATION IMPORTS (Metrics, visualization, analysis)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve, 
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from itertools import cycle

# 01_components
# dataset, batch_sampler, collate_function, 
# validation_dataset, validation_function, loss_function, model

# 02_training
# config_and_setup
# training_loop

# 03_evaluation
# testing_function
# testing_loop