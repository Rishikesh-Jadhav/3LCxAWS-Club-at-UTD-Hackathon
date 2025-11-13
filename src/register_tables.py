"""
3LC Table Registration Script
Hackathon: 3LC x AWS Cloud @ UT Dallas

This script registers your image dataset with 3LC for advanced analysis,
visualization, and data-centric AI workflows.

Usage:
    python register_tables.py
"""

import os
from pathlib import Path
from collections import Counter

def count_images(directory):
    """Count images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    count = 0
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in image_extensions:
                count += 1
    return count

def register_dataset():
    """
    Register the Chihuahua vs Muffin dataset with 3LC Tables
    
    This function:
    1. Scans the data directory structure
    2. Counts images per class
    3. Registers tables with 3LC (placeholder - requires 3LC installation)
    4. Prints summary statistics
    """
    
    print("=" * 60)
    print("3LC Dataset Registration - Chihuahua vs Muffin Challenge")
    print("=" * 60)
    
    # Define relative paths (works from project root)
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data"
    train_path = data_path / "train"
    test_path = data_path / "test"
    
    print(f"\n[Project Root]: {base_path}")
    print(f"[Data Path]: {data_path}")
    
    # Check if directories exist
    if not data_path.exists():
        print("\n[ERROR]: 'data/' directory not found!")
        print("Please download the dataset and extract it to the 'data/' folder.")
        return
    
    # Count images in training set
    print("\n[Training Set Analysis]:")
    print("-" * 40)
    
    chihuahua_path = train_path / "chihuahua"
    muffin_path = train_path / "muffin"
    
    chihuahua_count = count_images(chihuahua_path)
    muffin_count = count_images(muffin_path)
    
    print(f"  Chihuahua images: {chihuahua_count}")
    print(f"  Muffin images:    {muffin_count}")
    print(f"  Total training:   {chihuahua_count + muffin_count}")
    
    # Check class balance
    if chihuahua_count > 0 and muffin_count > 0:
        balance_ratio = min(chihuahua_count, muffin_count) / max(chihuahua_count, muffin_count)
        print(f"  Class balance:    {balance_ratio:.2%}")
        
        if balance_ratio < 0.9:
            print("  [WARNING]: Classes are imbalanced!")
        else:
            print("  [OK] Classes are well balanced")
    
    # Count test images
    if test_path.exists():
        test_count = count_images(test_path)
        print(f"\n[Test Set]: {test_count} images")
    else:
        print("\n[Test Set]: Not yet available")
    
    print("\n" + "=" * 60)
    print("3LC Table Registration")
    print("=" * 60)
    
    # Placeholder for actual 3LC registration
    # Uncomment and modify once 3LC is installed:
    """
    try:
        import tlc
        
        # Create a 3LC table from the training directory
        table = tlc.Table.from_image_folder(
            str(train_path),
            table_name="chihuahua_muffin_train"
        )
        
        print(f"[OK] Successfully registered training table: {table.url}")
        print(f"   Table ID: {table.table_id}")
        print(f"   Rows: {len(table)}")
        
        # Register test set if available
        if test_path.exists():
            test_table = tlc.Table.from_image_folder(
                str(test_path),
                table_name="chihuahua_muffin_test"
            )
            print(f"[OK] Successfully registered test table: {test_table.url}")
        
    except ImportError:
        print("[WARNING] 3LC not installed yet")
        print("   Install via: pip install tlc")
        print("   Sign up at: https://3lc.ai")
    except Exception as e:
        print(f"[ERROR] Error during 3LC registration: {e}")
    """
    
    print("\n[Next Steps]:")
    print("1. Install 3LC: pip install tlc")
    print("2. Sign up at https://3lc.ai and get your API key")
    print("3. Uncomment the 3LC registration code in this script")
    print("4. Run this script again to register your tables")
    print("5. Open 3LC Dashboard to explore your data!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    register_dataset()

