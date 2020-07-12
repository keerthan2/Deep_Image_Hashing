import numpy as np
from PIL import Image
import time
import copy

import pickle

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm_notebook as tqdm
import math
