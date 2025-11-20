# File: models/backbone_vit.py
# Simple ViT backbone wrapper for YOLOv5 integration with flexible input sizes.

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class SimpleViTBackbone(nn.Module):
    """
    Loads a timm ViT, extracts patch tokens, reshapes to a 2D feature map,
    and projects to `out_channels`. Handles variable input sizes by resizing.
    
    IMPORTANT: This module implements the YOLOv5 interface by wrapping output in Conv.

    Usage:
      from models.backbone_vit import SimpleViTBackbone
      backbone = SimpleViTBackbone(c1=3, c2=256, model_name='vit_tiny_patch16_224', pretrained=True)
      x = backbone(images)  # returns [B, c2, H, W]
    """
    def __init__(self, c1, c2, model_name='vit_tiny_patch16_224', pretrained=True):
        """
        Args:
            c1: Input channels (should be 3 for RGB images)
            c2: Output channels (will be projected to this)
            model_name: timm model name
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        # Create timm ViT - use standard 224 or specified size
        self.vit = timm.create_model(model_name, pretrained=pretrained, features_only=False)

        # timm ViT components
        assert hasattr(self.vit, 'patch_embed'), 'Unexpected vit model: missing patch_embed'
        self.patch_embed = self.vit.patch_embed
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        
        # Get patch size and expected input size
        self.patch_size = self.patch_embed.patch_size[0] if hasattr(self.patch_embed.patch_size, '__getitem__') else self.patch_embed.patch_size
        
        # Get the ViT's expected input size
        if hasattr(self.patch_embed, 'img_size'):
            vit_size = self.patch_embed.img_size
            self.vit_h = vit_size[0] if isinstance(vit_size, (tuple, list)) else vit_size
            self.vit_w = vit_size[1] if isinstance(vit_size, (tuple, list)) else vit_size
        else:
            # Default to 224 for standard ViT
            self.vit_h = self.vit_w = 224

        # Calculate output grid size for the ViT's native resolution
        self.native_grid_h = self.vit_h // self.patch_size
        self.native_grid_w = self.vit_w // self.patch_size

        # Embedding dimension
        if hasattr(self.patch_embed, 'proj'):
            embed_dim = self.patch_embed.proj.out_channels
        elif hasattr(self.vit, 'embed_dim'):
            embed_dim = self.vit.embed_dim
        else:
            embed_dim = 192  # default for vit_tiny

        # Project token dim -> desired conv channels (using nn.Conv2d so YOLOv5 can track it)
        self.project = nn.Conv2d(embed_dim, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        Forward pass that handles variable input sizes by resizing to ViT's expected size.
        
        Args:
            x: [B, C, H, W] input tensor
            
        Returns:
            feat: [B, c2, H', W'] feature map resized back to match input scale
        """
        B, C, H, W = x.shape
        
        # Calculate target output size (based on patch size)
        target_h = H // self.patch_size
        target_w = W // self.patch_size
        
        # Resize input to ViT's expected size if different
        if H != self.vit_h or W != self.vit_w:
            x_resized = F.interpolate(x, size=(self.vit_h, self.vit_w), mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # Patch embedding: returns [B, N, D]
        tokens = self.patch_embed(x_resized)
        
        # Pass through transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        
        # Remove cls token if present
        expected_patches = self.native_grid_h * self.native_grid_w
        if tokens.size(1) == expected_patches + 1:
            tokens = tokens[:, 1:, :]  # Remove CLS token
        
        B_t, N, D = tokens.shape
        
        # Reshape to 2D feature map at ViT's native grid size
        tokens = tokens.permute(0, 2, 1).contiguous().view(B_t, D, self.native_grid_h, self.native_grid_w)
        
        # Project to desired output channels (with BN and activation like Conv)
        feat = self.act(self.bn(self.project(tokens)))
        
        # Resize feature map to match the input's scale
        if feat.shape[2] != target_h or feat.shape[3] != target_w:
            feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return feat