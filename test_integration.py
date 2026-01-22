"""
Test script to verify the dataset-model integration.

Run this before starting training to ensure everything is properly configured.
"""

import sys
from pathlib import Path
import torch


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
        degradation_path = Path(degradation_path)
    else:
        # Auto-detect sibling folder
        degradation_path = Path(__file__).parent.parent / "Hardware-aware-degradation"
    
    if not degradation_path.exists():
        raise FileNotFoundError(
            f"Hardware-aware-degradation not found at: {degradation_path}\n"
            f"Please either:\n"
            f"  1. Ensure it exists as a sibling folder to BurstM\n"
            f"  2. Provide explicit path with --degradation_path argument"
        )
    
    # Add to Python path - add src/ FIRST so its utils package is found before BurstM's utils
    # This is critical because operators.py uses "from utils import bicubic_core"
    src_path = degradation_path / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Also add root for burst_dataset_wrapper import
    if str(degradation_path) not in sys.path:
        sys.path.insert(0, str(degradation_path))
    
    print(f"Using Hardware-aware-degradation from: {degradation_path}")
    
    return degradation_path


def test_dataset_loading(hr_image_dir, config_path, global_stats_path=None):
    """Test dataset creation and loading."""
    print("="*80)
    print("TEST 1: Dataset Loading")
    print("="*80)
    
    # Import after path is setup - use full package path
    from src.dataset import DegradationDataset
    
    # Check global_stats_path
    if global_stats_path and not Path(global_stats_path).exists():
        print(f"\n⚠ Warning: global_stats_path specified but file not found: {global_stats_path}")
        print(f"  Will use simple normalization instead.\n")
        global_stats_path = None
    elif global_stats_path is None:
        print(f"ℹ Note: No global_stats_path provided, using simple normalization.\n")
    
    try:
        # Create dataset
        dataset = DegradationDataset(
            hr_image_dir=hr_image_dir,
            config_path=config_path,
            global_stats_path=global_stats_path,
            augment=False,
            cache_size=10
        )
        print(f"✓ Dataset created successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Number of HR images: {len(dataset.hr_files)}")
        print(f"  - LR frames per sample: {dataset.num_lr_frames}")
        print(f"  - Downsampling factor: {dataset.downsampling_factor}")
        
        # Get one sample
        sample = dataset[0]
        print(f"\n✓ Sample loaded successfully")
        print(f"  - HR shape: {sample['hr'].shape}")
        print(f"  - Number of LR frames: {len(sample['lr'])}")
        print(f"  - LR frame shape: {sample['lr'][0].shape}")
        print(f"  - Flow vectors shape: {sample['flow_vectors'].shape}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        raise


def test_wrapper(dataset):
    """Test dataset wrapper."""
    print("\n" + "="*80)
    print("TEST 2: Dataset Wrapper")
    print("="*80)
    
    from burst_dataset_wrapper import BurstDatasetWrapper
    
    try:
        # Wrap dataset
        wrapped_dataset = BurstDatasetWrapper(dataset)
        print(f"✓ Wrapper created successfully")
        print(f"  - Number of samples: {len(wrapped_dataset)}")
        
        # Get one sample
        sample = wrapped_dataset[0]
        x, y, flow_vectors, meta_info, ds_factor, target_size = sample
        
        print(f"\n✓ Wrapped sample loaded successfully")
        print(f"  - x (burst) type: {type(x)}")
        print(f"  - x[0] (burst tensor) shape: {x[0].shape}")
        print(f"  - y (HR) shape: {y.shape}")
        print(f"  - flow_vectors shape: {flow_vectors.shape}")
        print(f"  - downsample_factor: {ds_factor}")
        print(f"  - target_size: {target_size}")
        print(f"  - meta_info keys: {list(meta_info.keys())}")
        
        return wrapped_dataset
        
    except Exception as e:
        print(f"✗ Wrapper failed: {e}")
        raise


def test_dataloader(wrapped_dataset, batch_size=2):
    """Test dataloader with collate function."""
    print("\n" + "="*80)
    print("TEST 3: DataLoader")
    print("="*80)
    
    from torch.utils.data import DataLoader
    from burst_dataset_wrapper import collate_fn
    
    try:
        # Create dataloader
        loader = DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        print(f"✓ DataLoader created successfully")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Number of batches: {len(loader)}")
        
        # Load one batch
        batch = next(iter(loader))
        x, y, flow_vectors, meta_info, ds_factor, target_size = batch
        
        print(f"\n✓ Batch loaded successfully")
        print(f"  - x[0] (burst batch) shape: {x[0].shape}")
        print(f"  - y (HR batch) shape: {y.shape}")
        print(f"  - flow_vectors batch shape: {flow_vectors.shape}")
        print(f"  - ds_factor batch shape: {ds_factor.shape}")
        print(f"  - target_size: {target_size}")
        print(f"  - meta_info length: {len(meta_info)}")
        
        return batch
        
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        raise


def test_model_forward(batch):
    """Test model forward pass."""
    print("\n" + "="*80)
    print("TEST 4: Model Forward Pass")
    print("="*80)
    
    from Network import BurstM
    
    try:
        # Create model
        model = BurstM(input_channels=1)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✓ Model loaded on CUDA")
        else:
            print(f"✓ Model loaded on CPU (CUDA not available)")
        
        # Unpack batch
        x, y, flow_vectors, meta_info, ds_factor, target_size = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            x = [x[0].cuda()]
            y = y.cuda()
            flow_vectors = flow_vectors.cuda()
            ds_factor = ds_factor.cuda()
        
        print(f"\n✓ Input prepared")
        print(f"  - Input burst shape: {x[0].shape}")
        print(f"  - Target HR shape: {y.shape}")
        
        # Forward pass
        with torch.no_grad():
            pred, ref, EstLrImg = model.forward(x, ds_factor[0].item(), target_size)
        
        print(f"\n✓ Forward pass successful")
        print(f"  - Prediction shape: {pred.shape}")
        print(f"  - Reference shape: {ref.shape}")
        print(f"  - Estimated LR shape: {EstLrImg.shape}")
        print(f"  - Prediction range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_training_step(batch):
    """Test a single training step."""
    print("\n" + "="*80)
    print("TEST 5: Training Step")
    print("="*80)
    
    from Network import BurstM
    
    try:
        # Create model
        model = BurstM(input_channels=1)
        model.train()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Unpack batch
        x, y, flow_vectors, meta_info, ds_factor, target_size = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            x = [x[0].cuda()]
            y = y.cuda()
            flow_vectors = flow_vectors.cuda()
            ds_factor = ds_factor.cuda()
        
        # Repack as tuple (as training_step expects)
        train_batch = (x, y, flow_vectors, meta_info, ds_factor, target_size)
        
        # Run training step
        loss = model.training_step(train_batch, 0)
        
        print(f"✓ Training step successful")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - Loss requires grad: {loss.requires_grad}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BurstM integration with custom dataset')
    parser.add_argument('--degradation_path', type=str, default=None,
                        help='Path to Hardware-aware-degradation repository (auto-detects if not provided)')
    parser.add_argument('--hr_image_dir', type=str, required=True,
                        help='Directory containing HR GeoTIFF images')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to degradation configuration YAML')
    parser.add_argument('--global_stats_path', type=str, default=None,
                        help='Path to global/combined percentile statistics YAML (e.g., global_stats.yaml or combined_stats.yaml). Optional.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    print("\n")
    print("="*80)
    print("BURSTM INTEGRATION TEST")
    print("="*80)
    
    # Setup degradation path
    print(f"\nSetting up paths...")
    try:
        degradation_repo_path = setup_degradation_path(args.degradation_path)
        print(f"✓ Hardware-aware-degradation found at: {degradation_repo_path}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return 1
    
    # Auto-detect stats file if not provided
    if args.global_stats_path is None:
        possible_stats = [
            degradation_repo_path / 'combined_stats.yaml',
            degradation_repo_path / 'global_stats.yaml',
        ]
        for stats_file in possible_stats:
            if stats_file.exists():
                args.global_stats_path = str(stats_file)
                print(f"✓ Auto-detected: {stats_file.name}\n")
                break
        else:
            print(f"ℹ No statistics file found (optional)\n")
    
    print(f"Configuration:")
    print(f"  HR images: {args.hr_image_dir}")
    print(f"  Config: {args.config_path}")
    print(f"  Global stats: {args.global_stats_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print()
    
    try:
        # Test 1: Dataset loading
        dataset = test_dataset_loading(
            args.hr_image_dir,
            args.config_path,
            args.global_stats_path
        )
        
        # Test 2: Wrapper
        wrapped_dataset = test_wrapper(dataset)
        
        # Test 3: DataLoader
        batch = test_dataloader(wrapped_dataset, args.batch_size)
        
        # Test 4: Model forward pass
        test_model_forward(batch)
        
        # Test 5: Training step
        test_training_step(batch)
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nYou can now run training with:")
        print(f"python train_custom_burst.py \\")
        print(f"    --hr_image_dir {args.hr_image_dir} \\")
        print(f"    --config_path {args.config_path} \\")
        if args.global_stats_path:
            print(f"    --global_stats_path {args.global_stats_path} \\")
        print(f"    --batch_size 1 \\")
        print(f"    --gpus 1")
        print()
        
    except Exception as e:
        print("\n" + "="*80)
        print("TESTS FAILED ✗")
        print("="*80)
        print(f"\nError: {e}")
        print("\nPlease fix the issues before running training.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
