"""=========================================================================================
ADVANCED DUAL-SR-BQNet: Next-Gen Satellite Image Super-Resolution System
=========================================================================================
HACKATHON-READY VERSION with all advanced features:
- Dual-input Swin Transformer architecture
- Robust multi-modal image registration
- Advanced training pipeline
- Comprehensive quality assessment
- Real-time web interface
- Support for both trained and untrained data scenarios
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import math
from datetime import datetime

# --- Advanced Imports ---
from piq import SSIMLoss, MultiScaleSSIMLoss, FID, LPIPS as VGGPerceptualLoss
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import streamlit as st
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import kornia
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from einops import rearrange, repeat
import timm
from timm.models.layers import DropPath, trunc_normal_
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imageio
import zipfile
from io import BytesIO
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple, List, Dict
from scipy import ndimage
import wandb

# Set random seeds for reproducibility
pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================================================================
# SECTION 0: ADVANCED SWIN TRANSFORMER MODULES WITH IMPROVEMENTS
# ===================================================================

class EnhancedMlp(nn.Module):
    """Enhanced MLP with residual connection and layer scaling"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., layer_scale_init=1e-6):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4  # Increased expansion ratio
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # Layer scaling
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(out_features)) if layer_scale_init > 0 else None
        
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        if self.gamma is not None:
            x = x * self.gamma
            
        return x + shortcut

class EnhancedWindowAttention(nn.Module):
    """Window attention with relative position bias and attention dropout"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 use_cos_attn=False, attn_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = attn_scale or head_dim ** -0.5
        self.use_cos_attn = use_cos_attn
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # Cosine attention
        if use_cos_attn:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
            
        # Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self._init_relative_position_index()
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def _init_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        if self.use_cos_attn:
            # Cosine attention
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1./0.01))).exp()
            attn = attn * logit_scale
        else:
            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EnhancedSwinTransformerBlock(nn.Module):
    """Enhanced Swin Transformer Block with multiple improvements"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 use_cos_attn=False, layer_scale_init=1e-6):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if min(input_resolution) > window_size else 0
        self.mlp_ratio = mlp_ratio
        
        # Enhanced components
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EnhancedWindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_cos_attn=use_cos_attn)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = EnhancedMlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), 
            act_layer=nn.GELU, drop=drop, layer_scale_init=layer_scale_init)
        
        # Attention mask for shifted windows
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None
            
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class EnhancedRSTB(nn.Module):
    """Residual Swin Transformer Block with enhanced features"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., use_cos_attn=False, layer_scale_init=1e-6, 
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Build blocks
        self.blocks = nn.ModuleList([
            EnhancedSwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                use_cos_attn=use_cos_attn, layer_scale_init=layer_scale_init)
            for i in range(depth)])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None
            
        # Convolutional residual block
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1))
        
    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"
        
        # Swin blocks
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                
        # Convolutional residual
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x) + x
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        
        # Downsample if needed
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x

# ===================================================================
# SECTION 1: ADVANCED IMAGE REGISTRATION & MODEL ARCHITECTURE
# ===================================================================

class MultiModalImageRegistration:
    """Robust multi-modal image registration with multiple feature detectors"""
    def __init__(self):
        # Initialize multiple feature detectors
        self.orb = cv2.ORB_create(2000)
        self.sift = cv2.SIFT_create()
        self.akaze = cv2.AKAZE_create()
        
        # FLANN parameters for different feature types
        # For ORB (binary features)
        FLANN_INDEX_LSH = 6
        orb_params = dict(algorithm=FLANN_INDEX_LSH,
                         table_number=6,
                         key_size=12,
                         multi_probe_level=1)
        self.flann_orb = cv2.FlannBasedMatcher(orb_params, dict(checks=50))
        
        # For SIFT/AKAZE (float features)
        FLANN_INDEX_KDTREE = 1
        sift_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.flann_sift = cv2.FlannBasedMatcher(sift_params, dict(checks=50))
        
        # Brute force matcher as fallback
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # RANSAC parameters
        self.ransac_thresh = 5.0
        self.min_matches = 10
        
    def _preprocess_image(self, img):
        """Convert to grayscale and normalize"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    def _extract_features(self, img):
        """Extract features using multiple methods with error handling"""
        img = self._preprocess_image(img)
        
        features = {
            'orb': (None, None),
            'sift': (None, None),
            'akaze': (None, None)
        }
        
        try:
            # ORB features
            kp_orb, des_orb = self.orb.detectAndCompute(img, None)
            if des_orb is not None:
                features['orb'] = (kp_orb, des_orb)
        except Exception as e:
            st.warning(f"ORB feature extraction failed: {str(e)}")
        
        try:
            # SIFT features
            kp_sift, des_sift = self.sift.detectAndCompute(img, None)
            if des_sift is not None:
                features['sift'] = (kp_sift, des_sift)
        except Exception as e:
            st.warning(f"SIFT feature extraction failed: {str(e)}")
        
        try:
            # AKAZE features
            kp_akaze, des_akaze = self.akaze.detectAndCompute(img, None)
            if des_akaze is not None:
                features['akaze'] = (kp_akaze, des_akaze)
        except Exception as e:
            st.warning(f"AKAZE feature extraction failed: {str(e)}")
        
        return features
    
    def _match_features(self, feat1, feat2):
        """Match features using multiple methods with error handling"""
        matches = {}
        
        # ORB matching
        if feat1['orb'][1] is not None and feat2['orb'][1] is not None:
            try:
                orb_matches = self.flann_orb.knnMatch(feat1['orb'][1], feat2['orb'][1], k=2)
                good_orb = []
                for m, n in orb_matches:
                    if m.distance < 0.7 * n.distance:
                        good_orb.append(m)
                matches['orb'] = good_orb
            except Exception as e:
                st.warning(f"ORB FLANN matching failed, using brute force: {str(e)}")
                try:
                    orb_matches = self.bf_matcher.match(feat1['orb'][1], feat2['orb'][1])
                    matches['orb'] = sorted(orb_matches, key=lambda x: x.distance)[:100]
                except Exception as e:
                    st.warning(f"ORB brute force matching failed: {str(e)}")
                    matches['orb'] = []
        
        # SIFT matching
        if feat1['sift'][1] is not None and feat2['sift'][1] is not None:
            try:
                sift_matches = self.flann_sift.knnMatch(feat1['sift'][1], feat2['sift'][1], k=2)
                good_sift = []
                for m, n in sift_matches:
                    if m.distance < 0.7 * n.distance:
                        good_sift.append(m)
                matches['sift'] = good_sift
            except Exception as e:
                st.warning(f"SIFT FLANN matching failed: {str(e)}")
                matches['sift'] = []
            
        # AKAZE matching (brute force)
        if feat1['akaze'][1] is not None and feat2['akaze'][1] is not None:
            try:
                akaze_matches = self.bf_matcher.match(feat1['akaze'][1], feat2['akaze'][1])
                matches['akaze'] = sorted(akaze_matches, key=lambda x: x.distance)[:100]
            except Exception as e:
                st.warning(f"AKAZE matching failed: {str(e)}")
                matches['akaze'] = []
            
        return matches
    
    def _estimate_homography(self, matches, feat1, feat2, method='orb'):
        """Estimate homography from matches with validation"""
        if len(matches.get(method, [])) < self.min_matches:
            return None
            
        try:
            src_pts = np.float32([feat1[method][0][m.queryIdx].pt for m in matches[method]]).reshape(-1,1,2)
            dst_pts = np.float32([feat2[method][0][m.trainIdx].pt for m in matches[method]]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
            
            # Validate homography
            if M is not None:
                det = np.linalg.det(M)
                if det < 0.1 or det > 10:  # Check for reasonable determinant
                    return None
                return M
            return None
        except Exception as e:
            st.warning(f"Homography estimation failed for {method}: {str(e)}")
            return None
    
    def register(self, ref_img, mov_img):
        """Register moving image to reference image using multi-modal features"""
        # Extract features from both images
        ref_feat = self._extract_features(ref_img)
        mov_feat = self._extract_features(mov_img)
        
        # Match features using all methods
        matches = self._match_features(mov_feat, ref_feat)
        
        # Try different methods in order of reliability
        for method in ['sift', 'akaze', 'orb']:
            if len(matches.get(method, [])) >= self.min_matches:
                M = self._estimate_homography(matches, mov_feat, ref_feat, method)
                if M is not None:
                    # Warp the moving image
                    h, w = ref_img.shape[:2]
                    registered = cv2.warpPerspective(mov_img, M, (w, h))
                    return registered
                    
        # Fallback: return original if no good registration
        st.warning("Image registration failed - using original image")
        return mov_img

class DualSwinIR(pl.LightningModule):
    """Advanced Dual-Input SwinIR model with enhanced features"""
    def __init__(self, img_size=64, patch_size=1, in_chans=1, embed_dim=64, 
                 depths=(4, 4, 4, 4), num_heads=(4, 4, 4, 4), window_size=8, 
                 scale=2, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, use_cos_attn=False, layer_scale_init=1e-6,
                 use_checkpoint=False):
        super().__init__()
        self.save_hyperparameters()
        
        # Input processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True))
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True))
            
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True))
            
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=embed_dim, embed_dim=embed_dim)
            
        # Absolute positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build RSTBs
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = EnhancedRSTB(
                dim=embed_dim,
                input_resolution=(img_size // patch_size, img_size // patch_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_cos_attn=use_cos_attn,
                layer_scale_init=layer_scale_init,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Reconstruction
        self.conv_after = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True))
            
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * scale ** 2, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, True))
            
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss(reduction='mean')
        self.ssim_loss = MultiScaleSSIMLoss(data_range=1.)
        self.fid = FID()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x1, x2):
        # Input processing
        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        
        # Feature fusion
        x = self.fusion(torch.cat([f1, f2], dim=1))
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        # Swin transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        # Normalization
        x = self.norm(x)
        
        # Reverse patch embedding
        x = self.patch_embed.reverse(x)
        
        # Reconstruction
        x = self.conv_after(x) + x
        x = self.upsample(x)
        x = self.conv_last(x)
        
        return x
        
    def training_step(self, batch, batch_idx):
        lr1, lr2, hr = batch
        sr = self(lr1, lr2)
        
        # Calculate losses
        l1_loss = self.l1_loss(sr, hr)
        vgg_loss = self.vgg_loss(sr, hr) * 0.1
        ssim_loss = self.ssim_loss(sr, hr)
        
        # Total loss
        loss = l1_loss + vgg_loss + ssim_loss
        
        # Log metrics
        self.log_dict({
            'train/loss': loss,
            'train/l1_loss': l1_loss,
            'train/vgg_loss': vgg_loss,
            'train/ssim_loss': ssim_loss
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        lr1, lr2, hr = batch
        sr = self(lr1, lr2)
        
        # Calculate metrics
        l1_loss = self.l1_loss(sr, hr)
        psnr_val = psnr(hr.cpu().numpy(), sr.detach().cpu().numpy(), data_range=1.0)
        ssim_val = ssim(hr.cpu().numpy(), sr.detach().cpu().numpy(), 
                       data_range=1.0, multichannel=True, win_size=7)
        
        # Log metrics
        self.log_dict({
            'val/l1_loss': l1_loss,
            'val/psnr': psnr_val,
            'val/ssim': ssim_val
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        # Visualize first image in batch
        if batch_idx == 0:
            self._log_images(lr1[0], lr2[0], hr[0], sr[0])
            
    def _log_images(self, lr1, lr2, hr, sr):
        """Log images to TensorBoard/WandB"""
        # Denormalize images
        lr1_img = (lr1.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
        lr2_img = (lr2.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
        hr_img = (hr.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
        sr_img = (sr.detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(lr1_img.squeeze(), cmap='gray')
        axes[0, 0].set_title('LR Input 1')
        axes[0, 1].imshow(lr2_img.squeeze(), cmap='gray')
        axes[0, 1].set_title('LR Input 2')
        axes[1, 0].imshow(hr_img.squeeze(), cmap='gray')
        axes[1, 0].set_title('HR Ground Truth')
        axes[1, 1].imshow(sr_img.squeeze(), cmap='gray')
        axes[1, 1].set_title('SR Output')
        
        for ax in axes.flat:
            ax.axis('off')
            
        plt.tight_layout()
        
        # Log to logger
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                "validation_images": wandb.Image(fig)
            })
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure("validation_images", fig, self.global_step)
            
        plt.close()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-4, 
            betas=(0.9, 0.999),
            weight_decay=0.01)
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=1e-6)
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

# ===================================================================
# SECTION 2: ADVANCED DATA PIPELINE
# ===================================================================

class AdvancedSatelliteSRDataset(Dataset):
    """Advanced dataset with multiple augmentations and degradation models"""
    def __init__(self, data_path, patch_size=128, scale=2, is_train=True, multi_frame=False):
        self.data_path = data_path
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train
        self.multi_frame = multi_frame
        
        # Get all image files
        self.image_files = []
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_files.append(os.path.join(root, f))
                    
        if not self.image_files:
            raise ValueError(f"No images found in {data_path}")
            
        # Define augmentations
        if is_train:
            self.augmentations = A.Compose([
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.OpticalDistortion(p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                A.ToFloat(max_value=255),
                ToTensorV2()
            ])
        else:
            self.augmentations = A.Compose([
                A.ToFloat(max_value=255),
                ToTensorV2()
            ])
            
        # Degradation kernels
        self.kernels = [
            np.ones((3, 3)) / 9,  # Uniform blur
            np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,  # Gaussian-like
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  # Sharpening
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity
        ]
        
    def __len__(self):
        return len(self.image_files) * (10 if self.is_train else 1)
        
    def _random_crop(self, img):
        """Random crop to patch size"""
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            img = cv2.resize(img, (max(self.patch_size, w), max(self.patch_size, h)), 
                            interpolation=cv2.INTER_CUBIC)
            
        h, w = img.shape[:2]
        y = np.random.randint(0, h - self.patch_size + 1)
        x = np.random.randint(0, w - self.patch_size + 1)
        return img[y:y + self.patch_size, x:x + self.patch_size]
        
    def _degrade_image(self, img):
        """Apply random degradation to image"""
        # Random kernel
        kernel = self.kernels[np.random.randint(0, len(self.kernels))]
        img = cv2.filter2D(img, -1, kernel)
        
        # Random noise
        if np.random.rand() > 0.5:
            sigma = np.random.uniform(1, 5)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 255)
            
        return img
        
    def __getitem__(self, idx):
        idx = idx % len(self.image_files)
        img_path = self.image_files[idx]
        
        # Load image
        img = np.array(Image.open(img_path).convert('L'))
        
        # Random crop
        img = self._random_crop(img)
        
        # Degrade for multi-frame scenario
        if self.multi_frame:
            img1 = self._degrade_image(img.copy())
            img2 = self._degrade_image(img.copy())
        else:
            img1 = img.copy()
            img2 = ndimage.shift(img.copy(), (0.5, 0.5), order=3)
            
        # Create LR images
        lr_size = (self.patch_size // self.scale, self.patch_size // self.scale)
        lr1 = cv2.resize(img1, lr_size, interpolation=cv2.INTER_AREA)
        lr2 = cv2.resize(img2, lr_size, interpolation=cv2.INTER_AREA)
        
        # Upscale LR images
        lr1 = cv2.resize(lr1, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        lr2 = cv2.resize(lr2, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        
        # Apply augmentations
        augmented = self.augmentations(image=img, lr1=lr1, lr2=lr2)
        hr = augmented['image']
        lr1 = augmented['lr1']
        lr2 = augmented['lr2']
        
        # Normalize to [-1, 1]
        hr = (hr - 0.5) / 0.5
        lr1 = (lr1 - 0.5) / 0.5
        lr2 = (lr2 - 0.5) / 0.5
        
        return lr1, lr2, hr

class AdvancedSatelliteDataModule(pl.LightningDataModule):
    """Advanced data module with distributed training support"""
    def __init__(self, data_path, batch_size=16, patch_size=128, scale=2, 
                 num_workers=4, multi_frame=False):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = AdvancedSatelliteSRDataset(
                self.hparams.data_path,
                patch_size=self.hparams.patch_size,
                scale=self.hparams.scale,
                is_train=True,
                multi_frame=self.hparams.multi_frame)
                
            # Split dataset
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size])
                
        if stage == 'test' or stage is None:
            self.test_dataset = AdvancedSatelliteSRDataset(
                self.hparams.data_path,
                patch_size=self.hparams.patch_size,
                scale=self.hparams.scale,
                is_train=False,
                multi_frame=self.hparams.multi_frame)
                
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0)
            
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0)
            
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0)

# ===================================================================
# SECTION 3: STREAMLIT APPLICATION WITH ADVANCED FEATURES
# ===================================================================

class SatelliteSuperResolutionApp:
    """Advanced Streamlit application for satellite image super-resolution"""
    def __init__(self):
        self.setup_page_config()
        self.model = None
        self.registration = MultiModalImageRegistration()
        self.classical_methods = {
            "Bicubic": cv2.INTER_CUBIC,
            "Lanczos": cv2.INTER_LANCZOS4,
            "Nearest": cv2.INTER_NEAREST,
            "Area": cv2.INTER_AREA
        }
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Advanced Satellite Image Super-Resolution",
            layout="wide",
            initial_sidebar_state="expanded")
            
        st.markdown("""
            <style>
            .main {padding: 2rem;}
            .sidebar .sidebar-content {padding: 1rem;}
            .stButton>button {width: 100%;}
            .stDownloadButton>button {width: 100%;}
            .stFileUploader>div {padding: 0.5rem;}
            .metric-card {border-radius: 0.5rem; padding: 1rem; background-color: #f0f2f6;}
            </style>
            """, unsafe_allow_html=True)
            
    def load_model(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Load the super-resolution model with error handling"""
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            st.info("Please train a model first or provide the correct path to a pretrained model")
            return None
            
        try:
            model = DualSwinIR(
                img_size=128,
                patch_size=1,
                in_chans=1,
                embed_dim=64,
                depths=[4, 4, 4, 4],
                num_heads=[4, 4, 4, 4],
                window_size=8,
                scale=2)
                
            # Handle both full model and state_dict loading
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except:
                # If loading state_dict fails, try loading the entire model
                model = torch.load(model_path, map_location=device)
                
            model.eval().to(device)
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("Please check if the model file is corrupted or incompatible")
            return None
            
    def denormalize(self, tensor):
        """Convert tensor from [-1, 1] to [0, 255]"""
        return ((tensor.clamp(-1, 1) + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
        
    def apply_classical_sr(self, img, scale=2, method='Bicubic'):
        """Apply classical super-resolution"""
        interpolation = self.classical_methods.get(method, cv2.INTER_CUBIC)
        return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=interpolation)
        
    def calculate_metrics(self, ref_img, test_img):
        """Calculate image quality metrics"""
        metrics = {}
        
        # PSNR
        metrics['PSNR'] = psnr(ref_img, test_img, data_range=255)
        
        # SSIM
        metrics['SSIM'] = ssim(ref_img, test_img, data_range=255, multichannel=False, win_size=7)
        
        return metrics
        
    def display_metrics(self, metrics):
        """Display metrics in a nice layout"""
        cols = st.columns(len(metrics))
        for i, (name, value) in enumerate(metrics.items()):
            if value is not None:
                cols[i].metric(
                    label=name,
                    value=f"{value:.2f}",
                    help=f"{name} (Higher is better)" if name in ['PSNR', 'SSIM'] else f"{name} (Lower is better)")
                    
    def plot_histograms(self, ref_img, test_img):
        """Plot intensity histograms for comparison"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        ax[0].hist(ref_img.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.5)
        ax[0].set_title('Reference Image Histogram')
        ax[0].set_xlabel('Pixel Intensity')
        ax[0].set_ylabel('Frequency')
        
        ax[1].hist(test_img.ravel(), bins=256, range=(0, 255), color='green', alpha=0.5)
        ax[1].set_title('SR Image Histogram')
        ax[1].set_xlabel('Pixel Intensity')
        
        st.pyplot(fig)
        plt.close()
        
    def train_model(self, data_path, epochs, batch_size, patch_size, output_path):
        """Train the super-resolution model"""
        try:
            # Setup data module
            dm = AdvancedSatelliteDataModule(
                data_path=data_path,
                batch_size=batch_size,
                patch_size=patch_size,
                scale=2,
                num_workers=4,
                multi_frame=True)
                
            # Initialize model
            model = DualSwinIR(
                img_size=patch_size,
                patch_size=1,
                in_chans=1,
                embed_dim=64,
                depths=[4, 4, 4, 4],
                num_heads=[4, 4, 4, 4],
                window_size=8,
                scale=2)
                
            # Setup callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath='checkpoints',
                filename='best-model-{epoch:02d}-{val_psnr:.2f}',
                monitor='val/psnr',
                mode='max',
                save_top_k=1)
                
            early_stopping = EarlyStopping(
                monitor='val/psnr',
                patience=10,
                mode='max')
                
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            
            # Setup logger
            wandb_logger = WandbLogger(project='satellite-sr', log_model=True)
            
            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=epochs,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                accelerator='auto',
                devices=1 if torch.cuda.is_available() else None,
                log_every_n_steps=10)
                
            # Train model
            with st.spinner("Training in progress..."):
                trainer.fit(model, dm)
                
            # Save final model
            torch.save(model.state_dict(), output_path)
            st.success(f"Training complete! Model saved to {output_path}")
            
            # Show best model path
            if checkpoint_callback.best_model_path:
                st.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
                # Offer to use the best model
                if st.button("Use best model"):
                    self.model = self.load_model(checkpoint_callback.best_model_path)
                
            return True
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return False
            
    def process_images(self, image1, image2, method, model_path=None, scale_factor=2):
        """Process input images and generate super-resolution output"""
        # Convert to numpy arrays
        img1 = np.array(image1.convert('L'))
        img2 = np.array(image2.convert('L'))
        
        # Register images
        with st.spinner("Registering images..."):
            try:
                img2_reg = self.registration.register(img1, img2)
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
                img2_reg = img2  # Fallback to original image
            
        # Display input images
        st.subheader("1. Input Images")
        col1, col2, col3 = st.columns(3)
        col1.image(img1, caption="Input Image 1", use_container_width=True)
        col2.image(img2, caption="Input Image 2", use_container_width=True)
        col3.image(img2_reg, caption="Registered Image 2", use_container_width=True)
        
        # Process based on selected method
        st.subheader("2. Super-Resolution Results")
        if method == "Deep Learning":
            if not self.model:
                self.model = self.load_model(model_path)
                
            if self.model:
                with st.spinner("Applying deep learning super-resolution..."):
                    try:
                        # Convert to tensor
                        transform = Compose([
                            ToTensor(),
                            Normalize(mean=[0.5], std=[0.5])
                        ])
                        
                        # Process both images
                        img1_t = transform(img1).unsqueeze(0).to(self.model.device)
                        img2_t = transform(img2_reg).unsqueeze(0).to(self.model.device)
                        
                        # Generate SR image
                        with torch.no_grad():
                            sr_t = self.model(img1_t, img2_t)
                            
                        # Convert back to numpy
                        sr_img = self.denormalize(sr_t.squeeze())
                    except Exception as e:
                        st.error(f"Deep learning SR failed: {str(e)}")
                        st.warning("Falling back to bicubic interpolation")
                        sr_img = self.apply_classical_sr(img1, scale_factor)
            else:
                st.error("Model not loaded. Using bicubic interpolation instead.")
                sr_img = self.apply_classical_sr(img1, scale_factor)
        else:
            with st.spinner(f"Applying {method} interpolation..."):
                sr_img = self.apply_classical_sr(img1, scale_factor, method)
                
        # Display results
        st.image(sr_img, caption=f"Super-Resolution Output ({method})", use_container_width=True)
        
        # Calculate metrics if we have a reference image
        st.subheader("3. Quality Assessment")
        if method == "Deep Learning":
            # For deep learning, compare with bicubic upscaling
            ref_img = self.apply_classical_sr(img1, scale_factor)
            metrics = self.calculate_metrics(ref_img, sr_img)
            self.display_metrics(metrics)
            
            # Show histogram comparison
            with st.expander("Intensity Histogram Comparison"):
                self.plot_histograms(ref_img, sr_img)
                
        # Add download button
        st.subheader("4. Download Results")
        buf = BytesIO()
        Image.fromarray(sr_img).save(buf, format="PNG")
        st.download_button(
            label="Download Super-Resolution Image",
            data=buf.getvalue(),
            file_name="sr_output.png",
            mime="image/png")
            
        return sr_img
        
    def run(self):
        """Run the Streamlit application"""
        st.title("🛰️ Advanced Satellite Image Super-Resolution System")
        st.markdown("""
            This application performs super-resolution on satellite images using either:
            - **Deep Learning**: Our advanced Dual-SwinIR model
            - **Classical Methods**: Traditional interpolation techniques
            """)
            
        with st.sidebar:
            st.header("Configuration")
            
            # Method selection
            sr_method = st.selectbox(
                "Super-Resolution Method",
                ["Deep Learning", "Bicubic", "Lanczos", "Nearest", "Area"])
                
            # Model options for deep learning
            if sr_method == "Deep Learning":
                model_path = st.text_input(
                    "Model Path",
                    value="models/dual_swinir.pth",
                    help="Path to pretrained model weights")
                    
                # Add button to manually load model
                if st.button("Load Model"):
                    self.model = self.load_model(model_path)
                    if self.model:
                        st.success("Model loaded successfully!")
                    
            # Scale factor
            scale_factor = st.slider(
                "Scale Factor",
                min_value=2,
                max_value=8,
                value=2,
                step=1)
                
            # Training section
            with st.expander("Train New Model", expanded=False):
                hr_folder = st.text_input(
                    "HR Training Data Folder",
                    value="training_data/hr",
                    help="Folder containing high-resolution training images")
                    
                epochs = st.number_input(
                    "Epochs",
                    min_value=1,
                    max_value=500,
                    value=50)
                    
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=64,
                    value=16)
                    
                patch_size = st.number_input(
                    "Patch Size",
                    min_value=32,
                    max_value=256,
                    value=128)
                    
                output_path = st.text_input(
                    "Output Model Path",
                    value="models/new_model.pth")
                    
                if st.button("Start Training"):
                    if os.path.isdir(hr_folder) and os.listdir(hr_folder):
                        self.train_model(hr_folder, epochs, batch_size, patch_size, output_path)
                    else:
                        st.error("Training data folder not found or is empty")
                        
            # Upload sample data
            with st.expander("Upload Sample Data", expanded=False):
                sample_zip = st.file_uploader(
                    "Upload sample data (ZIP)",
                    type=['zip'],
                    accept_multiple_files=False)
                    
                if sample_zip and st.button("Extract to Training Folder"):
                    with zipfile.ZipFile(sample_zip, 'r') as zip_ref:
                        zip_ref.extractall("training_data/hr")
                    st.success("Sample data extracted successfully!")
                    
        # Main content area
        st.header("Upload Satellite Images")
        col1, col2 = st.columns(2)
        image1 = col1.file_uploader(
            "Upload First Satellite Image",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
            
        image2 = col2.file_uploader(
            "Upload Second Satellite Image (Optional)",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
            
        if image1:
            try:
                if image2:
                    # Process dual images
                    self.process_images(
                        Image.open(image1),
                        Image.open(image2),
                        sr_method,
                        model_path if sr_method == "Deep Learning" else None,
                        scale_factor)
                else:
                    # Process single image (use shifted version as second input)
                    img = np.array(Image.open(image1).convert('L'))
                    img_shifted = ndimage.shift(img, (0.5, 0.5), order=3)
                    self.process_images(
                        Image.open(image1),
                        Image.fromarray(img_shifted),
                        sr_method,
                        model_path if sr_method == "Deep Learning" else None,
                        scale_factor)
            except Exception as e:
                st.error(f"Error processing images: {str(e)}")
                st.error("Please try with different images or check the console for details")
                    
        else:
            st.info("Please upload satellite images to begin processing")

# ===================================================================
# MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_data/hr", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Run the application
    app = SatelliteSuperResolutionApp()
    app.run()