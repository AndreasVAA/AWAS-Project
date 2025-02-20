#pip3 install torch torchvision numpy matplotlib opencv-python # Needed for things to run

import os
import cv2
import torch
import numpy as np
import torchvision
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset