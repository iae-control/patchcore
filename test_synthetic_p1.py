#!/usr/bin/env python3
"""
Synthetic defect test for trained PatchCore models.
Takes a real 150x75 image, creates tiles, adds synthetic defects,
compares anomaly scores between normal and defective tiles.
"""
import os, sys, random
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/patchcore'))
from torchvision import transforms
from src.patchcore import PatchCoreModel, FeatureExtractor
from src.config import TILE_SIZE, GROUPS