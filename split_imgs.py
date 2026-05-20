import random
import shutil
from pathlib import Path
from collections import defaultdict

def get_group_id(filename):
    """
    Extracts the group ID from the filename based on the first 5 components.
    Example: '74865_1014763_N-34-50-C-c-4-4_0_3125_mix.jpg' 
          -> '74865_1014763_N-34-50-C-c-4-4_0_3125'
    """
    base_name = Path(filename).stem
    parts = base_name.split('_')
    group_id = "_".join(parts[:5])
    
    return group_id

def split_grouped_dataset(
    images_dir, 
    labels_dir, 
    output_dir, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    random_seed=42,
    image_ext='.jpg',
    label_ext='.txt'
):
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # 1. Group images by their group ID
    groups = defaultdict(list)
    
    # Explicitly find ONLY the files matching the target image extension
    for img_file in images_path.glob(f'*{image_ext}'):
        if img_file.is_file():
            group_id = get_group_id(img_file.name)
            groups[group_id].append(img_file)
            
    group_ids = list(groups.keys())
    print(f"Found {len(group_ids)} unique groups containing {sum(len(v) for v in groups.values())} total {image_ext} images.")

    # 2. Shuffle groups
    random.shuffle(group_ids)

    # 3. Calculate split indices
    total_groups = len(group_ids)
    train_end = int(total_groups * train_ratio)
    val_end = train_end + int(total_groups * val_ratio)

    # 4. Assign groups to splits
    splits = {
        'train': group_ids[:train_end],
        'val': group_ids[train_end:val_end],
        'test': group_ids[val_end:]
    }

    # 5. Create output directories and copy files
    for split_name, assigned_groups in splits.items():
        print(f"\nProcessing '{split_name}' split ({len(assigned_groups)} groups)...")
        
        split_images_dir = output_path / split_name / 'images'
        split_labels_dir = output_path / split_name / 'labels'
        
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        missing_labels = 0

        # Copy files for each group assigned to this split
        for g_id in assigned_groups:
            for img_file in groups[g_id]:
                
                # Explicitly construct the exact expected label path (.txt)
                label_file = labels_path / f"{img_file.stem}{label_ext}"
                
                # Check if this exact file exists
                if label_file.exists():
                    shutil.copy2(img_file, split_images_dir / img_file.name)
                    shutil.copy2(label_file, split_labels_dir / label_file.name)
                else:
                    print(f"  Warning: No label found for {img_file.name} (Expected: {label_file.name})")
                    missing_labels += 1

        print(f"Finished '{split_name}'. Missing labels: {missing_labels}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGES_FOLDER = "./dataset_yolo_seg35/images"
    LABELS_FOLDER = "./dataset_yolo_seg35/labels"
    OUTPUT_FOLDER = "./dataset_split_all"
    
    # Explicitly set the extensions here
    IMAGE_EXTENSION = ".jpg"
    LABEL_EXTENSION = ".txt"
    
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Run the split
    split_grouped_dataset(
        images_dir=IMAGES_FOLDER,
        labels_dir=LABELS_FOLDER,
        output_dir=OUTPUT_FOLDER,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        image_ext=IMAGE_EXTENSION,
        label_ext=LABEL_EXTENSION
    )