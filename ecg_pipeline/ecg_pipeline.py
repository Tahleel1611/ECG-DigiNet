	def process_batch(self, image_paths: List[str], batch_size: int = 8, profile: bool = False) -> List[Dict]:
		"""
		Efficient batch processing of ECG images with optional profiling.
		Args:
			image_paths: List of image file paths
			batch_size: Number of images to process per batch (default 8)
			profile: If True, log timing for each batch
		Returns:
			List of result dicts
		"""
		import time
		results = []
		n = len(image_paths)
		pbar = tqdm(range(0, n, batch_size), desc='Batch ECG inference')
		for i in pbar:
			batch_paths = image_paths[i:i+batch_size]
			batch_results = []
			t0 = time.time() if profile else None
			for img_path in batch_paths:
				try:
					result = self.process_single_image(img_path)
					batch_results.append(result)
				except Exception as e:
					logger.error(f"Error processing {img_path}: {e}")
					batch_results.append({
						'image_path': img_path,
						'error': str(e)
					})
			t1 = time.time() if profile else None
			if profile:
				logger.info(f"Batch {i//batch_size+1}: {len(batch_paths)} images processed in {t1-t0:.2f} seconds")
			results.extend(batch_results)
		return results
	def train_epoch_profiled(self, train_loader: DataLoader, profile: bool = False) -> Dict:
		"""
		Train for one epoch with optional profiling (for large datasets).
		"""
		import time
		self.model.train()
		epoch_loss = 0.0
		loss_breakdown = {
			'seg_loss': 0.0,
			'bbox_loss': 0.0,
			'conf_loss': 0.0,
			'sig_loss': 0.0
		}
		pbar = tqdm(train_loader, desc='Training (profiled)', leave=False)
		for batch_idx, batch in enumerate(pbar):
			t0 = time.time() if profile else None
			images = batch['image'].to(self.device)
			targets = {
				'segmentation': batch.get('segmentation_mask', None),
				'bboxes': batch.get('bboxes', None),
				'confidences': batch.get('confidences', None),
				'signal_traces': batch.get('signal_traces', None)
			}
			targets = {k: v.to(self.device) if v is not None else None for k, v in targets.items()}
			self.optimizer.zero_grad()
			predictions = self.model(images)
			loss, loss_dict = self.criterion(predictions, targets)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			self.optimizer.step()
			epoch_loss += loss.item()
			for key in loss_breakdown:
				if key in loss_dict:
					loss_breakdown[key] += loss_dict[key]
			t1 = time.time() if profile else None
			if profile:
				logger.info(f"Batch {batch_idx+1}: {images.shape[0]} samples in {t1-t0:.2f} seconds, loss={loss.item():.4f}")
			pbar.set_postfix({'loss': loss.item()})
		n_batches = len(train_loader)
		epoch_loss /= n_batches
		for key in loss_breakdown:
			loss_breakdown[key] /= n_batches
		return {'epoch_loss': epoch_loss, **loss_breakdown}

# ...existing code...
# ECG Image to Time-Series Deep Learning Pipeline
# High-Performance 12-Lead ECG Signal Extraction with SNR Optimization

# Core Libraries
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import warnings
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Computer Vision
import cv2
from scipy import ndimage, interpolate, signal
from scipy.interpolate import CubicSpline, interp1d
from skimage import filters, morphology, measure, transform
from skimage.exposure import equalize_adapthist

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# Plotting and Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns

# Utilities
from tqdm import tqdm
import json
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

logger.info("All libraries imported successfully!")



