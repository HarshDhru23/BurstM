"""
Training script for BurstM using custom Hardware-aware Degradation Dataset.

This script bridges the Hardware-aware-degradation dataset with the BurstM model,
adapting the data format to match the expected inputs of Network.py.
"""

import os
import sys
import argparse
from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

# Import BurstM model (must be before dataset import)
from Network import BurstM

seed_everything(13)


def setup_degradation_path(degradation_path=None):
    """
    Setup path to Hardware-aware-degradation repository.
    
    Args:
        degradation_path: Optional override path. If None, auto-detects sibling folder.
        
    Returns:
        Path to Hardware-aware-degradation
        
    Raises:
        FileNotFoundError: If path cannot be found
    """
    if degradation_path is not None:
        # User provided explicit path
        degradation_path = Path(degradation_path)
    else:
        # Auto-detect: assume sibling folder structure
        # /scratch/user/BurstM/ -> /scratch/user/Hardware-aware-degradation/
        degradation_path = Path(__file__).parent.parent / "Hardware-aware-degradation"
    
    if not degradation_path.exists():
        raise FileNotFoundError(
            f"Hardware-aware-degradation not found at: {degradation_path}\n"
            f"Please either:\n"
            f"  1. Ensure it exists as a sibling folder to BurstM\n"
            f"  2. Provide explicit path with --degradation_path argument"
        )
    
    # Add to Python path - insert at beginning to avoid conflicts
    if str(degradation_path) not in sys.path:
        sys.path.insert(0, str(degradation_path))
    
    # Also add the src directory explicitly to avoid import conflicts
    src_path = degradation_path / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print(f"Using Hardware-aware-degradation from: {degradation_path}")
    
    return degradation_path


def torch_seed(random_seed=13):
    """Set random seeds for reproducibility."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train BurstM with Hardware-aware Degradation Dataset'
    )
    
    # Path parameters
    parser.add_argument('--degradation_path', type=str, default=None,
                        help='Path to Hardware-aware-degradation repository (auto-detects if not provided)')
    
    # Data parameters
    parser.add_argument('--hr_image_dir', type=str, required=True,
                        help='Directory containing HR GeoTIFF images')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to degradation configuration YAML')
    parser.add_argument('--global_stats_path', type=str, default=None,
                        help='Path to global/combined percentile statistics YAML (e.g., global_stats.yaml or combined_stats.yaml). Optional - uses simple normalization if not provided.')
    
    # Dataset parameters
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Fraction of data for training (rest for validation)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable 8-way augmentation')
    parser.add_argument('--no_augment', dest='augment', action='store_false',
                        help='Disable augmentation')
    parser.add_argument('--cache_size', type=int, default=100,
                        help='Number of HR images to cache in memory')
    parser.add_argument('--seed', type=int, default=13,
                        help='Random seed for reproducibility')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (BurstM typically uses 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='Maximum number of training epochs')
    parser.add_argument('--val_check_interval', type=float, default=0.25,
                        help='Validation check interval (fraction of epoch)')
    
    # Model checkpoint parameters
    parser.add_argument('--model_dir', type=str, default='./Results/CustomBurst/saved_model',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./Results/CustomBurst/tensorboard',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Hardware parameters
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32],
                        help='Training precision (16 for mixed precision, 32 for full)')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu', 'ddp', 'dp'],
                        help='Accelerator type for training')
    
    return parser.parse_args()


def create_datasets(args):
    """
    Create training and validation datasets.
    
    Args:
        args: Command line arguments
        
    Returns:
        train_dataset, val_dataset: Wrapped datasets ready for DataLoader
    """
    # Check global_stats_path with auto-detection
    if args.global_stats_path is None:
        # Try to auto-detect combined_stats.yaml or global_stats.yaml in degradation repo
        possible_stats_files = [
            degradation_repo_path / 'combined_stats.yaml',
            degradation_repo_path / 'global_stats.yaml',
        ]
        for stats_file in possible_stats_files:
            if stats_file.exists():
                args.global_stats_path = str(stats_file)
                print(f"\n✓ Auto-detected statistics file: {stats_file.name}")
                break
    
    if args.global_stats_path and not Path(args.global_stats_path).exists():
        print(f"\n⚠ Warning: Statistics file specified but not found: {args.global_stats_path}")
        print(f"  Will use simple normalization (divide by 65535) instead.")
        print(f"  To generate statistics, run:")
        print(f"    python compute_global_stats.py --input_dir {args.hr_image_dir} --output global_stats.yaml")
        print(f"  Or combine multiple datasets:")
        print(f"    python combine_histograms.py --histograms hist1.npz hist2.npz --output combined_stats.yaml\n")
        args.global_stats_path = None
    elif args.global_stats_path is None:
        print(f"\n⚠ Note: No statistics file found (looked for combined_stats.yaml and global_stats.yaml).")
        print(f"  Using simple normalization (divide by 65535).")
        print(f"  For better results, generate statistics first.\n")
    else:
        # Load and display stats info
        stats_path = Path(args.global_stats_path)
        with open(stats_path, 'r') as f:
            import yaml
            stats = yaml.safe_load(f)
        print(f"\n✓ Using statistics file: {stats_path.name}")
        print(f"  - p2 (2nd percentile): {stats.get('p2', 'N/A'):.2f}")
        print(f"  - p98 (98th percentile): {stats.get('p98', 'N/A'):.2f}")
        if 'metadata' in stats and 'num_datasets' in stats['metadata']:
            print(f"  - Combined from {stats['metadata']['num_datasets']} datasets")
        if 'suggested_normalization_factor' in stats:
            print(f"  - Suggested normalization: {stats['suggested_normalization_factor']:.2f}")
        print()
    
    # Create full degradation dataset
    DegradationDataset = args._DegradationDataset
    BurstDatasetWrapper = args._BurstDatasetWrapper
    
    full_dataset = DegradationDataset(
        hr_image_dir=args.hr_image_dir,
        config_path=args.config_path,
        global_stats_path=args.global_stats_path,
        augment=args.augment,
        cache_size=args.cache_size,
        seed=args.seed
    )
    
    # Split into train/val based on number of unique HR images (before augmentation)
    num_hr_images = len(full_dataset.hr_files)
    num_train = int(num_hr_images * args.train_split)
    num_val = num_hr_images - num_train
    
    print(f"\nDataset Split:")
    print(f"  Total HR images: {num_hr_images}")
    print(f"  Training images: {num_train}")
    print(f"  Validation images: {num_val}")
    
    # Calculate indices accounting for augmentations
    num_augmentations = len(full_dataset.augmentations)
    train_indices = []
    val_indices = []
    
    for img_idx in range(num_hr_images):
        base_idx = img_idx * num_augmentations
        indices = list(range(base_idx, base_idx + num_augmentations))
        
        if img_idx < num_train:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)
    
    print(f"  Training samples (with augmentation): {len(train_indices)}")
    print(f"  Validation samples (with augmentation): {len(val_indices)}")
    
    # Create subsets
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Wrap datasets for BurstM format
    train_dataset = BurstDatasetWrapper(train_subset)
    val_dataset = BurstDatasetWrapper(val_subset)
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, args):
    """
    Create training and validation dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        args: Command line arguments
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    collate_fn = args._collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nDataLoaders:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def setup_training(args):
    """
    Setup model, callbacks, and trainer.
    
    Args:
        args: Command line arguments
        
    Returns:
        model, trainer: BurstM model and PyTorch Lightning Trainer
    """
    # Initialize model for grayscale (1-channel) input
    torch_seed(args.seed)
    model = BurstM(input_channels=1)  # Use 1 channel for grayscale
    
    # Load from checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"\nLoading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',
        dirpath=args.model_dir,
        filename='burstm-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=3,
        save_last=True,
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name='custom_burst_training',
        version=None,
        default_hp_metric=False
    )
    
    # Create trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.gpus if args.accelerator in ['gpu', 'auto'] else None,
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=0.01,
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=args.val_check_interval,
        logger=tb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return model, trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("="*80)
    print("BurstM Training with Hardware-aware Degradation Dataset")
    print("="*80)
    
    # Setup path to Hardware-aware-degradation repository
    print("\nSetting up paths...")
    degradation_repo_path = setup_degradation_path(args.degradation_path)
    
    # Import dataset modules (must be after path setup)
    try:
        # Import from dataset module (now src is in path)
        from dataset import DegradationDataset
        
        # Import wrapper from Hardware-aware-degradation root
        sys.path.insert(0, str(degradation_repo_path))
        from burst_dataset_wrapper import BurstDatasetWrapper, collate_fn
        
        print("✓ Successfully imported dataset modules")
    except ImportError as e:
        print(f"✗ Failed to import dataset modules: {e}")
        print(f"\nDebug information:")
        print(f"  sys.path: {sys.path[:3]}")
        print(f"  degradation_repo_path: {degradation_repo_path}")
        print(f"\nPlease ensure Hardware-aware-degradation repository is set up correctly.")
        import traceback
        traceback.print_exc()
        return 1
    
    # Store imports in args for use by other functions
    args._DegradationDataset = DegradationDataset
    args._BurstDatasetWrapper = BurstDatasetWrapper
    args._collate_fn = collate_fn
    
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        if not arg.startswith('_'):
            print(f"  {arg}: {value}")
    
    # Create datasets
    print("\n" + "="*80)
    print("Creating Datasets...")
    print("="*80)
    train_dataset, val_dataset = create_datasets(args)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args)
    
    # Setup training
    print("\n" + "="*80)
    print("Setting up Training...")
    print("="*80)
    model, trainer = setup_training(args)
    
    # Start training
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Model checkpoints saved to: {args.model_dir}")
    print(f"TensorBoard logs saved to: {args.log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
