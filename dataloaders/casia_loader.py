import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random


class CASIADataset(Dataset):
    """CASIA 2.0 Dataset for Copy-Move Forgery Detection"""
    def __init__(self, data_dir, split='train', transform=None, image_size=(512, 512)):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Dataset structure
        self.authentic_dir = os.path.join(data_dir, 'Au')
        self.tampered_dir = os.path.join(data_dir, 'Tp')
        self.mask_dir = os.path.join(data_dir, 'Gt')
        
        # Get file lists
        self.authentic_files = self._get_file_list(self.authentic_dir)
        self.tampered_files = self._get_file_list(self.tampered_dir)
        
        # Create train/test split
        self.train_files, self.test_files = self._create_split()
        
        # Select appropriate split
        if split == 'train':
            self.files = self.train_files
        else:
            self.files = self.test_files
        
        print(f"Loaded {len(self.files)} files for {split} split")
    
    def _get_file_list(self, directory):
        """Get list of image files from directory"""
        if not os.path.exists(directory):
            return []
        
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']:
            files.extend([f for f in os.listdir(directory) if f.lower().endswith(ext[1:])])
        
        return sorted(files)
    
    def _create_split(self, train_ratio=0.8):
        """Create train/test split"""
        # Combine authentic and tampered files
        all_files = []
        
        # Add authentic files
        for file in self.authentic_files:
            all_files.append({
                'type': 'authentic',
                'file': file,
                'image_path': os.path.join(self.authentic_dir, file),
                'mask_path': None
            })
        
        # Add tampered files
        for file in self.tampered_files:
            mask_file = file.replace('.jpg', '_gt.png').replace('.jpeg', '_gt.png').replace('.png', '_gt.png')
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            all_files.append({
                'type': 'tampered',
                'file': file,
                'image_path': os.path.join(self.tampered_dir, file),
                'mask_path': mask_path if os.path.exists(mask_path) else None
            })
        
        # Shuffle and split
        random.shuffle(all_files)
        split_idx = int(len(all_files) * train_ratio)
        
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]
        
        return train_files, test_files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_info = self.files[idx]
        
        # Load image
        image = self._load_image(file_info['image_path'])
        
        # Load mask if available
        if file_info['mask_path'] and os.path.exists(file_info['mask_path']):
            mask = self._load_mask(file_info['mask_path'])
        else:
            # Create empty mask for authentic images
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default transformations
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image,
            'mask': mask,
            'type': file_info['type'],
            'file': file_info['file']
        }
    
    def _load_image(self, image_path):
        """Load image from path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((*self.image_size, 3), dtype=np.uint8)
    
    def _load_mask(self, mask_path):
        """Load mask from path"""
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            return mask
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            # Return empty mask as fallback
            return np.zeros(self.image_size, dtype=np.uint8)


def get_transforms(config, split='train'):
    """Get data transformations"""
    image_size = tuple(config['dataset']['image_size'])
    
    if split == 'train':
        # Training transformations with augmentation
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=config['augmentation']['horizontal_flip']),
            A.VerticalFlip(p=config['augmentation']['vertical_flip']),
            A.Rotate(limit=config['augmentation']['rotation'], p=0.5),
            A.ColorJitter(
                brightness=config['augmentation']['brightness'],
                contrast=config['augmentation']['contrast'],
                saturation=config['augmentation']['saturation'],
                hue=config['augmentation']['hue'],
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        # Validation/Test transformations
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    
    return transform


class CASIADataLoader:
    """Data loader factory for CASIA dataset"""
    def __init__(self, config):
        self.config = config
        self.data_dir = config['paths']['data_dir']
        
    def get_train_loader(self):
        """Get training data loader"""
        transform = get_transforms(self.config, 'train')
        dataset = CASIADataset(
            data_dir=self.data_dir,
            split='train',
            transform=transform,
            image_size=tuple(self.config['dataset']['image_size'])
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_loader(self):
        """Get validation data loader"""
        transform = get_transforms(self.config, 'val')
        dataset = CASIADataset(
            data_dir=self.data_dir,
            split='test',  # Use test split for validation
            transform=transform,
            image_size=tuple(self.config['dataset']['image_size'])
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
    
    def get_test_loader(self):
        """Get test data loader"""
        transform = get_transforms(self.config, 'test')
        dataset = CASIADataset(
            data_dir=self.data_dir,
            split='test',
            transform=transform,
            image_size=tuple(self.config['dataset']['image_size'])
        )
        
        return DataLoader(
            dataset,
            batch_size=1,  # Test with batch size 1
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        ) 