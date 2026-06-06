import os
import glob
import json
from PIL import Image, ImageDraw

def extract_real_panels_with_geometry(real_img_dir, real_lbl_dir, output_panel_dir):
    os.makedirs(output_panel_dir, exist_ok=True)
    panel_counter = 0
    img_files = glob.glob(os.path.join(real_img_dir, "*.*"))
    
    for img_path in img_files:
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(real_lbl_dir, f"{base_name}.txt")
        
        if not os.path.exists(lbl_path): continue
            
        with Image.open(img_path).convert("RGBA") as img:
            w, h = img.size
            
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 3: continue
                    
                    coords = parts[1:]
                    xs = [x * w for x in coords[0::2]]
                    ys = [y * h for y in coords[1::2]]
                    
                    # 1. Create an exact stencil mask
                    mask = Image.new("L", (w, h), 0)
                    ImageDraw.Draw(mask).polygon(list(zip(xs, ys)), fill=255)
                    
                    # 2. Paste onto a guaranteed transparent canvas
                    transparent_bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    transparent_bg.paste(img, (0, 0), mask=mask)
                    
                    # 3. Crop to bounds to save disk space
                    min_x, min_y = max(0, int(min(xs))), max(0, int(min(ys)))
                    max_x, max_y = min(w, int(max(xs))), min(h, int(max(ys)))
                    if max_x <= min_x or max_y <= min_y: continue
                    
                    final_panel = transparent_bg.crop((min_x, min_y, max_x, max_y))
                    
                    # 4. Save Image
                    out_name = f"extracted_{base_name}_{panel_counter}"
                    final_panel.save(os.path.join(output_panel_dir, f"{out_name}.png"), "PNG")
                    
                    # 5. Calculate & Save Relative Normalized Polygon
                    panel_w = max_x - min_x
                    panel_h = max_y - min_y
                    rel_poly = [((x - min_x) / panel_w, (y - min_y) / panel_h) for x, y in zip(xs, ys)]
                    
                    with open(os.path.join(output_panel_dir, f"{out_name}.json"), "w") as jf:
                        json.dump({"polygon": rel_poly}, jf)
                        
                    panel_counter += 1
                    
    print(f"Extracted {panel_counter} perfectly masked panels with geometry.")

extract_real_panels_with_geometry("real_pv_gda.yolov8-obb70/train/images", "real_pv_gda.yolov8-obb70/train/labels", "./extracted_panels2")
