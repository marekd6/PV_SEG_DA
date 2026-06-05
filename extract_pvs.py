import os
import glob
from PIL import Image, ImageDraw

def extract_real_panels(real_img_dir, real_lbl_dir, output_panel_dir):
    os.makedirs(output_panel_dir, exist_ok=True)
    panel_counter = 0
    
    img_files = glob.glob(os.path.join(real_img_dir, "*.*"))
    
    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(real_lbl_dir, f"{base_name}.txt")
        
        if not os.path.exists(lbl_path):
            continue
            
        with Image.open(img_path).convert("RGBA") as img:
            w, h = img.size
            
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 3: continue
                    
                    # YOLO format: class x1 y1 x2 y2 ... (normalized)
                    coords = parts[1:]
                    xs = [x * w for x in coords[0::2]]
                    ys = [y * h for y in coords[1::2]]
                    
                    # Get bounding box for cropping
                    min_x, min_y = int(min(xs)), int(min(ys))
                    max_x, max_y = int(max(xs)), int(max(ys))
                    
                    # Create a mask for the exact polygon shape
                    mask = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(mask)
                    xy_pairs = list(zip(xs, ys))
                    draw.polygon(xy_pairs, fill=255)
                    
                    # Crop image and mask to the bounding box
                    bbox = (min_x, min_y, max_x, max_y)
                    panel_crop = img.crop(bbox)
                    mask_crop = mask.crop(bbox)
                    
                    # Apply mask as alpha channel
                    panel_crop.putalpha(mask_crop)
                    
                    # Save extracted panel
                    out_name = f"extracted_{base_name}_{panel_counter}.png"
                    panel_crop.save(os.path.join(output_panel_dir, out_name))
                    panel_counter += 1
                    
    print(f"Extracted {panel_counter} real panels.")

extract_real_panels("real_pv_gda.yolov8-obb70/train/images", "real_pv_gda.yolov8-obb70/train/labels", "./extracted_panels")
