#!/usr/bin/env python3
"""
AMSR2 Sequential Trainer - GPU Optimized Edition with Memory Management
Sequential file processing with aggressive memory management for 8x super-resolution

Author: Volodymyr Didur
Version: 6.0 - Memory-Safe GPU Sequential Processing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import time
import json
import argparse
from pathlib import Path
import psutil
import sys
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# GPU-optimized thread settings
if torch.cuda.is_available():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Important for memory management
    torch.cuda.empty_cache()
else:
    torch.set_num_threads(min(8, os.cpu_count()))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('amsr2_gpu_sequential.log')
    ]
)
logger = logging.getLogger(__name__)


# ====== AGGRESSIVE MEMORY MANAGEMENT ======
def aggressive_cleanup():
    """Aggressive memory cleanup for GPU and CPU"""
    # Python garbage collection
    gc.collect()

    # PyTorch GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force cleanup multiple times
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()


def log_memory_usage(prefix=""):
    """Log current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(f"{prefix} GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # System memory
    memory = psutil.virtual_memory()
    logger.info(
        f"{prefix} RAM: {memory.percent:.1f}% used ({memory.used / 1024 ** 3:.1f}GB / {memory.total / 1024 ** 3:.1f}GB)")


# ====== LIGHTWEIGHT DATASET FOR SINGLE FILE ======
class SingleFileAMSR2Dataset(Dataset):
    """Lightweight dataset that processes one NPZ file without keeping all data in memory"""

    def __init__(self, npz_path: str, preprocessor,
                 degradation_scale: int = 8,  # Changed from 4 to 8
                 augment: bool = True,
                 filter_orbit_type: Optional[str] = None,
                 max_swaths_in_memory: int = 100):  # Limit swaths in memory

        self.npz_path = npz_path
        self.preprocessor = preprocessor
        self.degradation_scale = degradation_scale
        self.augment = augment
        self.filter_orbit_type = filter_orbit_type
        self.max_swaths_in_memory = max_swaths_in_memory

        # Only store indices, not data
        self.valid_indices = self._scan_file()

    def _scan_file(self):
        """Scan file to get valid swath indices without loading data"""
        logger.info(f"📂 Scanning file: {os.path.basename(self.npz_path)}")
        valid_indices = []
        rejected_count = 0

        try:
            with np.load(self.npz_path, allow_pickle=True) as data:
                if 'swath_array' not in data:
                    logger.error(f"❌ Invalid file structure: {self.npz_path}")
                    return []

                swath_array = data['swath_array']

                for idx in range(len(swath_array)):
                    try:
                        swath_dict = swath_array[idx]
                        swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                        if 'temperature' not in swath or 'metadata' not in swath:
                            rejected_count += 1
                            continue

                        # Filter by orbit type
                        if self.filter_orbit_type is not None:
                            orbit_type = swath['metadata'].get('orbit_type', 'U')
                            if orbit_type != self.filter_orbit_type:
                                rejected_count += 1
                                continue

                        # Дополнительная проверка - загружаем температуру для валидации
                        temperature = swath['temperature']
                        metadata = swath['metadata']

                        # Применяем scale factor
                        scale_factor = metadata.get('scale_factor', 1.0)
                        if temperature.dtype != np.float32:
                            temperature = temperature.astype(np.float32) * scale_factor

                        # Проверяем на negative strides
                        if any(s < 0 for s in temperature.strides):
                            logger.warning(f"Swath {idx}: negative strides detected, skipping")
                            rejected_count += 1
                            continue

                        # Фильтруем невалидные значения
                        temperature = np.where(temperature < 50, np.nan, temperature)
                        temperature = np.where(temperature > 350, np.nan, temperature)

                        # Проверяем валидность
                        valid_pixels = np.sum(~np.isnan(temperature))
                        total_pixels = temperature.size
                        validity_ratio = valid_pixels / total_pixels

                        if validity_ratio < 0.5:  # Более строгий порог - минимум 50% валидных пикселей
                            logger.debug(f"Swath {idx}: rejected, only {validity_ratio:.1%} valid pixels")
                            rejected_count += 1
                            continue

                        # Если прошли все проверки - добавляем
                        valid_indices.append(idx)

                    except Exception as e:
                        logger.warning(f"Swath {idx}: error during validation: {e}")
                        rejected_count += 1
                        continue

                logger.info(f"✅ Found {len(valid_indices)} valid swaths, rejected {rejected_count}")

                # Ограничение количества swaths
                if len(valid_indices) > self.max_swaths_in_memory:
                    logger.warning(f"⚠️ Limiting to {self.max_swaths_in_memory} swaths to save memory")
                    valid_indices = valid_indices[:self.max_swaths_in_memory]

                return valid_indices

        except Exception as e:
            logger.error(f"❌ Error scanning file {self.npz_path}: {e}")
            return []

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Load and process a single swath on demand"""
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for {len(self.valid_indices)} valid swaths")

        swath_idx = self.valid_indices[idx]

        try:
            with np.load(self.npz_path, allow_pickle=True) as data:
                swath_array = data['swath_array']
                swath_dict = swath_array[swath_idx]
                swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                # Process temperature
                raw_temperature = swath['temperature']
                metadata = swath['metadata']

                # Apply scale factor
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = raw_temperature.astype(np.float32) * scale_factor

                # Fix negative strides if any
                if any(s < 0 for s in temperature.strides):
                    temperature = temperature.copy()

                # Clear raw data
                del raw_temperature

                # Filter invalid values
                temperature = np.where(temperature < 50, np.nan, temperature)
                temperature = np.where(temperature > 350, np.nan, temperature)

                # Process data
                temperature = self.preprocessor.crop_and_pad_to_target(temperature)
                temperature = self.preprocessor.normalize_brightness_temperature(temperature)

                if self.augment:
                    temperature = self._augment_data(temperature)

                # Create degraded version
                degraded = self._create_degradation(temperature)

                high_res = torch.from_numpy(temperature).unsqueeze(0).float()
                low_res = torch.from_numpy(degraded).unsqueeze(0).float()

                del temperature, degraded

                return low_res, high_res

        except Exception as e:
            logger.error(f"❌ Unexpected error loading swath {swath_idx}: {e}")
            raise  # Пусть падает - мы уже проверили все swaths при сканировании

    def _create_degradation(self, high_res: np.ndarray) -> np.ndarray:
        """Create degraded version for 8x super-resolution training"""
        h, w = high_res.shape

        # Convert to tensor for GPU operations if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        high_res_tensor = torch.from_numpy(high_res).unsqueeze(0).unsqueeze(0).float().to(device)

        # Downsample by 8x
        low_res = F.interpolate(
            high_res_tensor,
            size=(h // self.degradation_scale, w // self.degradation_scale),
            mode='bilinear',
            align_corners=False
        )

        # Add realistic noise
        noise = torch.randn_like(low_res) * 0.01
        low_res = low_res + noise

        # Apply slight blur (sensor PSF)
        if low_res.shape[-1] > 3 and low_res.shape[-2] > 3:
            blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
            low_res = blur(low_res)

        # НЕ ДЕЛАЕМ UPSAMPLE ОБРАТНО!
        # Модель сама увеличит в 8 раз

        result = low_res.squeeze().cpu().numpy()

        # Clean up GPU memory
        del high_res_tensor, low_res
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return result

    def _augment_data(self, data: np.ndarray) -> np.ndarray:
        """Simple augmentation"""
        if not self.augment or np.random.rand() > 0.3:
            return data

        if np.random.rand() > 0.5:
            data = np.fliplr(data)
        if np.random.rand() > 0.5:
            data = np.flipud(data)

        return data


# ====== PREPROCESSOR ======
class AMSR2NPZDataPreprocessor:
    """Preprocessor for AMSR2 data"""

    def __init__(self, target_height: int = 2048, target_width: int = 208):  # Changed to 200
        self.target_height = target_height
        self.target_width = target_width
        logger.info(f"📏 Preprocessor configured for size: {target_height}x{target_width}")

    def crop_and_pad_to_target(self, temperature: np.ndarray) -> np.ndarray:
        """Crop or pad to target size"""
        h, w = temperature.shape

        # Crop if larger
        if h > self.target_height:
            start_h = (h - self.target_height) // 2
            temperature = temperature[start_h:start_h + self.target_height]

        if w > self.target_width:
            start_w = (w - self.target_width) // 2
            temperature = temperature[:, start_w:start_w + self.target_width]

        current_h, current_w = temperature.shape

        # Pad if smaller
        if current_h < self.target_height or current_w < self.target_width:
            pad_h = max(0, self.target_height - current_h)
            pad_w = max(0, self.target_width - current_w)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            temperature = np.pad(temperature,
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='reflect')

        return temperature

    def normalize_brightness_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Normalize brightness temperature"""
        valid_mask = ~np.isnan(temperature)
        if np.sum(valid_mask) > 0:
            mean_temp = np.mean(temperature[valid_mask])
            temperature = np.where(np.isnan(temperature), mean_temp, temperature)
        else:
            temperature = np.full_like(temperature, 250.0)

        temperature = np.clip(temperature, 50, 350)
        normalized = (temperature - 200) / 150

        return normalized.astype(np.float32)


# ====== MODEL ARCHITECTURE (Keep as is) ======
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UNetResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = F.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return x, features


class UNetDecoder(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.up4 = self._make_upconv_block(512, 256)
        self.up3 = self._make_upconv_block(256 + 256, 128)
        self.up2 = self._make_upconv_block(128 + 128, 64)
        self.up1 = self._make_upconv_block(64 + 64, 64)
        self.final_up = nn.ConvTranspose2d(64 + 64, 32, 2, 2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1)
        )

    def _make_upconv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        print(f"Decoder input shape: {x.shape}")
        for i, feat in enumerate(skip_features):
            print(f"Skip feature {i} shape: {feat.shape}")

        x = self.up4(x)
        print(f"After up4: {x.shape}, skip_features[3]: {skip_features[3].shape}")
        if x.shape[2] != skip_features[3].shape[2] or x.shape[3] != skip_features[3].shape[3]:
            diff_h = skip_features[3].shape[2] - x.shape[2]
            diff_w = skip_features[3].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[3]], dim=1)

        x = self.up3(x)
        print(f"After up3: {x.shape}, skip_features[2]: {skip_features[2].shape}")
        if x.shape[2] != skip_features[2].shape[2] or x.shape[3] != skip_features[2].shape[3]:
            diff_h = skip_features[2].shape[2] - x.shape[2]
            diff_w = skip_features[2].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[2]], dim=1)

        x = self.up2(x)
        print(f"After up2: {x.shape}, skip_features[1]: {skip_features[1].shape}")
        if x.shape[2] != skip_features[1].shape[2] or x.shape[3] != skip_features[1].shape[3]:
            diff_h = skip_features[1].shape[2] - x.shape[2]
            diff_w = skip_features[1].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[1]], dim=1)

        x = self.up1(x)
        print(f"After up1: {x.shape}, skip_features[0]: {skip_features[0].shape}")
        if x.shape[2] != skip_features[0].shape[2] or x.shape[3] != skip_features[0].shape[3]:
            diff_h = skip_features[0].shape[2] - x.shape[2]
            diff_w = skip_features[0].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[0]], dim=1)

        x = self.final_up(x)
        x = self.final_conv(x)
        return x


class UNetResNetSuperResolution(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, scale_factor: int = 8, target_height: int = 2048, target_width: int = 200):  # Changed to 8
        super().__init__()
        self.scale_factor = scale_factor
        self.target_height = target_height
        self.target_width = target_width
        self.encoder = UNetResNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

        # Progressive upsampling for 8x
        if scale_factor > 1:
            upsampling_layers = []
            current_scale = 1

            while current_scale < scale_factor:
                if scale_factor // current_scale >= 4:
                    factor = 4
                elif scale_factor // current_scale >= 2:
                    factor = 2
                else:
                    factor = scale_factor // current_scale

                upsampling_layers.extend([
                    nn.ConvTranspose2d(out_channels if len(upsampling_layers) == 0 else 32, 32,
                                       factor, factor),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(inplace=True)
                ])
                current_scale *= factor

            upsampling_layers.append(nn.Conv2d(32, out_channels, 1))
            self.upsampling = nn.Sequential(*upsampling_layers)
        else:
            self.upsampling = nn.Identity()

    def forward(self, x):
        encoded, skip_features = self.encoder(x)
        decoded = self.decoder(encoded, skip_features)
        output = self.upsampling(decoded)

        if hasattr(self, 'target_height') and hasattr(self, 'target_width'):
            if output.shape[2] > self.target_height:
                output = output[:, :, :self.target_height, :]
            if output.shape[3] > self.target_width:
                output = output[:, :, :, :self.target_width]

        return output


# ====== LOSS FUNCTION ======
class AMSR2SpecificLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.15, gamma: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def compute_gradients(x):
            grad_x = x[:, :, :-1, :] - x[:, :, 1:, :]
            grad_y = x[:, :, :, :-1] - x[:, :, :, 1:]
            return grad_x, grad_y

        pred_grad_x, pred_grad_y = compute_gradients(pred)
        target_grad_x, target_grad_y = compute_gradients(target)

        loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        loss_y = self.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def brightness_temperature_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        energy_loss = self.mse_loss(pred_mean, target_mean)

        pred_std = torch.std(pred, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        distribution_loss = self.mse_loss(pred_std, target_std)

        range_penalty = torch.mean(torch.relu(torch.abs(pred) - 1.0))

        return energy_loss + 0.5 * distribution_loss + 0.1 * range_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        l1_loss = self.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        phys_loss = self.brightness_temperature_consistency(pred, target)

        total_loss = (self.alpha * l1_loss +
                      self.beta * grad_loss +
                      self.gamma * phys_loss)

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'gradient_loss': grad_loss.item(),
            'physical_loss': phys_loss.item(),
            'total_loss': total_loss.item()
        }


# ====== MEMORY-SAFE SEQUENTIAL TRAINER ======
class MemorySafeSequentialTrainer:
    """GPU-optimized sequential trainer with aggressive memory management"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5,
                 use_amp: bool = True, gradient_accumulation_steps: int = 1):

        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.criterion = AMSR2SpecificLoss()
        self.training_history = []
        self.best_loss = float('inf')

    def train_on_file(self, file_path: str, preprocessor,
                      epochs_per_file: int = 10,  # Train 10 epochs per file
                      batch_size: int = 4,
                      augment: bool = True,
                      filter_orbit_type: Optional[str] = None):
        """Train on one file with memory management"""

        logger.info(f"📚 Training on file: {os.path.basename(file_path)}")
        log_memory_usage("Before loading file:")

        # Aggressive cleanup before starting
        aggressive_cleanup()

        file_results = []

        for epoch in range(epochs_per_file):
            logger.info(f"   Epoch {epoch + 1}/{epochs_per_file}")

            # Create dataset for one file - it will be garbage collected after each epoch
            dataset = SingleFileAMSR2Dataset(
                npz_path=file_path,
                preprocessor=preprocessor,
                degradation_scale=8,  # 8x super-resolution
                augment=augment,
                filter_orbit_type=filter_orbit_type,
                max_swaths_in_memory=1000  # Limit swaths
            )

            if len(dataset) == 0:
                logger.warning(f"⚠️ Empty file, skipping: {file_path}")
                return {'loss': float('inf'), 'swaths': 0}

            # DataLoader with minimal workers to save memory
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False,
                persistent_workers=False,  # Don't keep workers alive
                drop_last=True
            )

            self.model.train()
            epoch_losses = []

            try:
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)

                for batch_idx, (low_res, high_res) in enumerate(progress_bar):
                    low_res = low_res.to(self.device, non_blocking=True)
                    high_res = high_res.to(self.device, non_blocking=True)

                    # Mixed precision training
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pred = self.model(low_res)
                        loss, loss_components = self.criterion(pred, high_res)
                        loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient accumulation
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()

                        self.optimizer.zero_grad()

                    epoch_losses.append(loss.item() * self.gradient_accumulation_steps)

                    # Update progress bar
                    progress_bar.set_postfix({'loss': f'{epoch_losses[-1]:.4f}'})

                    # Aggressive memory cleanup every 50 batches
                    if batch_idx % 50 == 0:
                        del low_res, high_res, pred, loss
                        aggressive_cleanup()

                # Handle remaining gradients
                if len(dataloader) % self.gradient_accumulation_steps != 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("💾 GPU out of memory! Attempting recovery...")
                    aggressive_cleanup()
                    return {'loss': float('inf'), 'swaths': 0}
                else:
                    raise e

            finally:
                # Clean up after each epoch
                del dataset, dataloader
                aggressive_cleanup()

            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            file_results.append({'epoch': epoch + 1, 'loss': avg_epoch_loss})

            logger.info(f"   Epoch {epoch + 1} loss: {avg_epoch_loss:.4f}")
            log_memory_usage(f"   After epoch {epoch + 1}:")

        # Calculate file statistics
        avg_file_loss = np.mean([r['loss'] for r in file_results if r['loss'] != float('inf')])

        return {
            'loss': avg_file_loss,
            'swaths': len(dataset) if 'dataset' in locals() else 0,
            'epochs_trained': epochs_per_file,
            'epoch_results': file_results
        }

    def train_sequential(self, npz_files: List[str], preprocessor,
                         epochs_per_file: int = 10,
                         batch_size: int = 4,
                         save_path: str = "best_amsr2_model.pth"):
        """Sequential training on all files with memory management"""

        logger.info(f"🚀 Starting memory-safe sequential training:")
        logger.info(f"   Files: {len(npz_files)}")
        logger.info(f"   Epochs per file: {epochs_per_file}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Device: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"   Mixed precision: {self.use_amp}")
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        total_files = len(npz_files)
        processed_files = 0

        # Initial cleanup
        aggressive_cleanup()
        log_memory_usage("Initial state:")

        for file_idx, file_path in enumerate(npz_files):
            logger.info(f"\n📂 File {file_idx + 1}/{total_files}: {os.path.basename(file_path)}")

            # Train on this file for multiple epochs
            result = self.train_on_file(
                file_path=file_path,
                preprocessor=preprocessor,
                epochs_per_file=epochs_per_file,
                batch_size=batch_size,
                augment=True,
                filter_orbit_type=None
            )

            # Save best model
            if result['loss'] < self.best_loss and result['loss'] != float('inf'):
                self.best_loss = result['loss']

                # Save checkpoint
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_loss,
                    'file_idx': file_idx,
                    'scale_factor': 8,  # Important: save the scale factor
                    'device': str(self.device)
                }

                torch.save(checkpoint, save_path)
                logger.info(f"💾 Saved best model: loss={self.best_loss:.4f}")

            # Update training history
            self.training_history.append({
                'file_idx': file_idx,
                'file_path': file_path,
                'avg_loss': result['loss'],
                'total_swaths': result.get('swaths', 0),
                'epochs': epochs_per_file
            })

            processed_files += 1

            logger.info(f"📊 File completed: avg_loss={result['loss']:.4f}")
            logger.info(f"   Progress: {processed_files}/{total_files} files")

            # Scheduler step
            if result['loss'] != float('inf'):
                self.scheduler.step(result['loss'])

            # Aggressive cleanup after each file
            aggressive_cleanup()
            log_memory_usage("After file cleanup:")

        logger.info(f"\n🎉 Sequential training completed!")
        logger.info(f"   Files processed: {processed_files}/{total_files}")
        logger.info(f"   Best loss: {self.best_loss:.4f}")

        return self.training_history


# ====== UTILITY FUNCTIONS ======
def find_npz_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Find NPZ files with optional limit"""

    if not os.path.exists(directory):
        logger.error(f"❌ Directory does not exist: {directory}")
        return []

    pattern = os.path.join(directory, "*.npz")
    all_files = glob.glob(pattern)

    if not all_files:
        logger.error(f"❌ No NPZ files found in directory: {directory}")
        return []

    # Sort files for reproducibility
    all_files.sort()

    if max_files is not None and max_files > 0:
        selected_files = all_files[:max_files]
        logger.info(f"📁 Found {len(all_files)} NPZ files, selected {len(selected_files)}")
    else:
        selected_files = all_files
        logger.info(f"📁 Found {len(selected_files)} NPZ files")

    # Check file sizes
    total_size_gb = 0
    for file_path in selected_files:
        size_gb = os.path.getsize(file_path) / 1024 ** 3
        total_size_gb += size_gb

    logger.info(f"📊 Total data size: {total_size_gb:.2f} GB")

    return selected_files


def test_model_inference(model_path: str, test_file: str, preprocessor, device):
    """Test the trained model on a new file"""

    logger.info("🧪 Testing model inference...")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    model = UNetResNetSuperResolution(
        in_channels=1,
        out_channels=1,
        scale_factor=8  # 8x super-resolution
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create test dataset
    dataset = SingleFileAMSR2Dataset(
        npz_path=test_file,
        preprocessor=preprocessor,
        degradation_scale=8,
        augment=False,
        max_swaths_in_memory=10  # Just a few for testing
    )

    if len(dataset) == 0:
        logger.error("❌ No valid swaths in test file")
        return

    # Test on first swath
    with torch.no_grad():
        low_res, high_res = dataset[0]
        low_res = low_res.unsqueeze(0).to(device)
        high_res = high_res.unsqueeze(0).to(device)

        # Inference
        start_time = time.time()
        pred = model(low_res)
        inference_time = time.time() - start_time

        logger.info(f"✅ Inference successful!")
        logger.info(f"   Input shape: {low_res.shape}")
        logger.info(f"   Output shape: {pred.shape}")
        logger.info(f"   Expected shape: {high_res.shape}")
        logger.info(f"   Inference time: {inference_time:.3f}s")

        # Check if 8x scaling worked
        if pred.shape[-1] == high_res.shape[-1] and pred.shape[-2] == high_res.shape[-2]:
            logger.info("✅ 8x super-resolution confirmed!")

        # For 2000x200 input, output should be 16000x1600
        logger.info(f"📏 For 2000x200 input → output will be 16000x1600")


def create_training_summary(training_history: List[Dict], save_path: str = "training_summary_gpu.json"):
    """Create training summary"""

    if not training_history:
        return

    valid_results = [h for h in training_history if h['avg_loss'] != float('inf')]

    if not valid_results:
        logger.warning("⚠️ No valid results for summary")
        return

    summary = {
        'total_files_processed': len(training_history),
        'successful_files': len(valid_results),
        'total_swaths': sum(h['total_swaths'] for h in valid_results),
        'best_loss': min(h['avg_loss'] for h in valid_results),
        'worst_loss': max(h['avg_loss'] for h in valid_results),
        'average_loss': np.mean([h['avg_loss'] for h in valid_results]),
        'training_history': training_history,
        'scale_factor': 8,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"📄 Summary saved: {save_path}")


# ====== MAIN FUNCTION ======
def main():
    """Main function for memory-safe GPU sequential training"""

    parser = argparse.ArgumentParser(
        description='AMSR2 Memory-Safe GPU Sequential 8x Super-Resolution Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:

1. Train on first 10 files:
   python gpu_sequential_amsr2_fixed.py --npz-dir /path/to/data --max-files 10

2. Train with specific batch size:
   python gpu_sequential_amsr2_fixed.py --npz-dir /path/to/data --batch-size 2

3. Quick test on 5 files:
   python gpu_sequential_amsr2_fixed.py --npz-dir /path/to/data --max-files 5 --epochs-per-file 5
        '''
    )

    # Required parameters
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Path to directory with NPZ files')

    # Data parameters
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to train on (default: all)')
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Filter by orbit type')

    # Training parameters
    parser.add_argument('--epochs-per-file', type=int, default=10,
                        help='Number of epochs per file (default: 10)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2 for memory safety)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='Gradient accumulation steps (default: 2)')

    # System parameters
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True for GPU)')
    parser.add_argument('--target-height', type=int, default=2000,
                        help='Target height for AMSR2 (default: 2000)')
    parser.add_argument('--target-width', type=int, default=200,
                        help='Target width for AMSR2 (default: 200)')

    # Output parameters
    parser.add_argument('--save-path', type=str, default='best_amsr2_8x_model.pth',
                        help='Path to save best model')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Optional: test file for inference after training')

    args = parser.parse_args()

    print("🛰️ AMSR2 MEMORY-SAFE 8x SUPER-RESOLUTION TRAINER")
    print("=" * 60)

    # Device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # Initial cleanup
        aggressive_cleanup()

        # Log current memory usage
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        logger.info(f"   Currently allocated: {allocated:.1f} GB")
    else:
        logger.warning("⚠️ No GPU available, using CPU")
        logger.warning("   Training will be much slower")
        args.use_amp = False

    # Memory check
    memory_info = psutil.virtual_memory()
    logger.info(f"💾 System RAM: {memory_info.available / 1024 ** 3:.1f} GB available "
                f"of {memory_info.total / 1024 ** 3:.1f} GB ({memory_info.percent:.1f}% used)")

    # Find NPZ files
    npz_files = find_npz_files(args.npz_dir, args.max_files)

    if not npz_files:
        logger.error("❌ No NPZ files found")
        sys.exit(1)

    # Create model
    logger.info("🧠 Creating 8x super-resolution model...")
    model = UNetResNetSuperResolution(
        in_channels=1,
        out_channels=1,
        scale_factor=8  # 8x super-resolution
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parameters: {total_params:,}")
    logger.info(f"   Model size: {total_params * 4 / 1024 ** 2:.1f} MB")
    logger.info(f"   Scale factor: 8x")

    # Create preprocessor
    preprocessor = AMSR2NPZDataPreprocessor(
        target_height=args.target_height,
        target_width=args.target_width
    )

    # Create trainer
    trainer = MemorySafeSequentialTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )

    # Configuration summary
    logger.info(f"⚙️ Training configuration:")
    logger.info(f"   Files to train: {len(npz_files)}")
    logger.info(f"   Epochs per file: {args.epochs_per_file}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"   Effective batch size: {args.batch_size * args.gradient_accumulation}")
    logger.info(f"   Learning rate: {args.lr}")
    logger.info(f"   Target size: {args.target_height}x{args.target_width}")
    logger.info(f"   Output size: {args.target_height * 8}x{args.target_width * 8}")
    logger.info(f"   Mixed precision: {args.use_amp}")

    # Start training
    logger.info("\n🚀 Starting memory-safe sequential training...")
    start_time = time.time()

    try:
        training_history = trainer.train_sequential(
            npz_files=npz_files,
            preprocessor=preprocessor,
            epochs_per_file=args.epochs_per_file,
            batch_size=args.batch_size,
            save_path=args.save_path
        )

        end_time = time.time()
        training_time = end_time - start_time

        logger.info(f"\n🎉 Training completed!")
        logger.info(f"   Training time: {training_time / 3600:.2f} hours")
        logger.info(f"   Files processed: {len(training_history)}")

        # Create summary
        create_training_summary(training_history)

        # Test inference if requested
        if args.test_file and os.path.exists(args.test_file):
            test_model_inference(args.save_path, args.test_file, preprocessor, device)

        logger.info(f"\n📁 Results:")
        logger.info(f"   Model: {args.save_path}")
        logger.info(f"   Summary: training_summary_gpu.json")
        logger.info(f"   Log: amsr2_gpu_sequential.log")

    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("💾 GPU out of memory!")
            logger.error("   Try reducing batch_size or increasing gradient_accumulation")
        else:
            logger.error(f"❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if device.type == 'cuda':
            aggressive_cleanup()
        logger.info("🛑 Program finished")


if __name__ == "__main__":
    main()