'''
draws and saves masks from labels as .png files
'''

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def yolo_seg_to_segformer(yolo_dir: str, output_dir: str, bg_index: int = 0, class_id int = 255):
    """
    Converts a YOLO instance segmentation dataset to SegFormer semantic segmentation format.
    
    Args:
        yolo_dir (str): Path to the root of the YOLO dataset (containing images/ and labels/).
        output_dir (str): Path where the SegFormer dataset will be saved.
        bg_index (int): The pixel value for the background.
    """
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)

    # Standard splits in YOLO
    splits = ['train', 'val', 'test']
    splits = ['train', 'val']
    splits = ['test']

    for split in splits:
        img_dir = yolo_dir / split / 'images'
        label_dir = yolo_dir / split / 'labels'
        
        if not img_dir.exists():
            continue
            
        # Create output directories
        out_img_dir = output_dir / split / 'images'
        out_ann_dir = output_dir / split / 'annotations'
        
        # out_img_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)

        images = list(img_dir.glob('*.*'))
        
        for img_path in tqdm(images, desc=f"Processing {split} split"):
            # 1. Copy original image to the new directory
            # shutil.copy(img_path, out_img_dir / img_path.name)
            
            # 2. Read image dimensions (needed to denormalize YOLO coordinates)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping.")
                continue
            h, w = img.shape[:2]
            
            # 3. Initialize the semantic mask with the background index
            mask = np.full((h, w), bg_index, dtype=np.uint8)
            
            label_path = label_dir / f"{img_path.stem}.txt"
            
            # 4. Parse YOLO label file and draw polygons
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    
                    # YOLO instance segmentation format: class_id x1 y1 x2 y2 ... xn yn
                    if len(parts) < 7: 
                        continue # Needs at least class_id + 3 points (x, y) to make a polygon
                                       
                    # Extract polygon coordinates and reshape to (N, 2)
                    coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                    
                    # Denormalize coordinates
                    coords[:, 0] = coords[:, 0] * w
                    coords[:, 1] = coords[:, 1] * h
                    coords = np.int32(coords)
                    
                    # Draw filled polygon on the mask using the class_id as the pixel value
                    cv2.fillPoly(mask, [coords], class_id)
            
            # 5. Save the mask as a PNG (SegFormer requires image-like annotations)
            mask_filename = f"{img_path.stem}.png"
            cv2.imwrite(str(out_ann_dir / mask_filename), mask)

if __name__ == "__main__":

    INPUT_YOLO_DIR = "/users/project1/pt01299/synt/gda" 
    
    OUTPUT_SEGFORMER_DIR = "/users/project1/pt01299/synt/gda" 
    
    print("Starting conversion...")
    yolo_seg_to_segformer(INPUT_YOLO_DIR, OUTPUT_SEGFORMER_DIR, bg_index=0, class_id=255)
    print("Conversion complete. Dataset is ready for SegFormer.")
