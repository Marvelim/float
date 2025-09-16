"""
Loss functions for LIA-X training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import torchvision.transforms as transforms


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    def __init__(self, layers=['conv_1_2', 'conv_2_2', 'conv_3_3', 'conv_4_3']):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features
        self.layer_name_mapping = {
            '3': 'conv_1_2',
            '8': 'conv_2_2', 
            '15': 'conv_3_3',
            '22': 'conv_4_3'
        }
        self.layers = layers
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        loss = 0
        x_feats = self.extract_features(x)
        y_feats = self.extract_features(y)
        
        for layer in self.layers:
            if layer in x_feats and layer in y_feats:
                loss += F.mse_loss(x_feats[layer], y_feats[layer])
                
        return loss
    
    def extract_features(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_name_mapping:
                features[self.layer_name_mapping[name]] = x
        return features


class IdentityLoss(nn.Module):
    """Identity preservation loss."""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, recon, target):
        return self.l1_loss(recon, target)


class MotionConsistencyLoss(nn.Module):
    """Loss to ensure motion parameter consistency."""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_motion, target_motion):
        return self.mse_loss(pred_motion, target_motion)


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video sequences."""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, frames):
        """
        Args:
            frames: Tensor of shape [B, T, C, H, W]
        """
        if frames.size(1) < 2:
            return torch.tensor(0.0, device=frames.device)
            
        loss = 0
        for i in range(frames.size(1) - 1):
            diff = frames[:, i+1] - frames[:, i]
            loss += self.l1_loss(diff, torch.zeros_like(diff))
            
        return loss / (frames.size(1) - 1)


class LIAXLoss(nn.Module):
    """Combined loss for LIA-X training."""
    def __init__(self, 
                 recon_weight=1.0,
                 perceptual_weight=1.0, 
                 identity_weight=10.0,
                 motion_weight=1.0,
                 temporal_weight=0.1):
        super().__init__()
        
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.identity_weight = identity_weight
        self.motion_weight = motion_weight
        self.temporal_weight = temporal_weight
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.identity_loss = IdentityLoss() 
        self.motion_loss = MotionConsistencyLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict containing model outputs
                - 'recon': Reconstructed images [B, C, H, W]
                - 'motion': Predicted motion parameters [B, motion_dim]
                - 'frames': Generated video frames [B, T, C, H, W] (optional)
            targets: Dict containing ground truth
                - 'images': Target images [B, C, H, W]
                - 'motion': Target motion parameters [B, motion_dim]
                - 'identity': Identity images [B, C, H, W] (optional)
        """
        total_loss = 0
        losses = {}
        
        # Reconstruction loss
        if 'recon' in predictions and 'images' in targets:
            recon_loss = self.l1_loss(predictions['recon'], targets['images'])
            losses['reconstruction'] = recon_loss
            total_loss += self.recon_weight * recon_loss
            
        # Perceptual loss
        if 'recon' in predictions and 'images' in targets:
            try:
                perceptual_loss = self.perceptual_loss(predictions['recon'], targets['images'])
                losses['perceptual'] = perceptual_loss
                total_loss += self.perceptual_weight * perceptual_loss
            except:
                # Skip if VGG fails
                pass
                
        # Identity preservation loss
        if 'recon' in predictions and 'identity' in targets:
            identity_loss = self.identity_loss(predictions['recon'], targets['identity'])
            losses['identity'] = identity_loss
            total_loss += self.identity_weight * identity_loss
            
        # Motion consistency loss
        if 'motion' in predictions and 'motion' in targets:
            motion_loss = self.motion_loss(predictions['motion'], targets['motion'])
            losses['motion'] = motion_loss
            total_loss += self.motion_weight * motion_loss
            
        # Temporal consistency loss
        if 'frames' in predictions:
            temporal_loss = self.temporal_loss(predictions['frames'])
            losses['temporal'] = temporal_loss
            total_loss += self.temporal_weight * temporal_loss
            
        losses['total'] = total_loss
        return total_loss, losses


def create_loss_function(config):
    """Create loss function from configuration."""
    return LIAXLoss(
        recon_weight=config.get('recon_weight', 1.0),
        perceptual_weight=config.get('perceptual_weight', 1.0),
        identity_weight=config.get('identity_weight', 10.0),
        motion_weight=config.get('motion_weight', 1.0),
        temporal_weight=config.get('temporal_weight', 0.1)
    )