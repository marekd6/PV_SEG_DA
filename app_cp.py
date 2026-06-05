import random
import json
import os
import glob
from PIL import Image

# Directories
DEFAULT_BG_FOLDER = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26v4'
DEFAULT_PANEL_FOLDER = './pvs'

BG_DIR = DEFAULT_BG_FOLDER # "dataset_yolo_seg35/rzeczywiste" # Using real images as backgrounds too
EXTRACTED_PANELS_DIR = "./extracted_panels"
LBL_DIR = "dataset_yolo_seg35/labels"
META_DIR = "dataset_yolo_seg35/meta"
OUTPUT_DIR = "dataset_synthetic_real"
OUTPUT_META = os.path.join(OUTPUT_DIR, "meta")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_META, exist_ok=True)

def apply_geometry(img, rot, stretch_x, stretch_y):
    res = img.copy()
    if stretch_x != 1.0 or stretch_y != 1.0:
        new_w = max(1, int(res.width * stretch_x))
        new_h = max(1, int(res.height * stretch_y))
        res = res.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return res.rotate(rot, expand=True, resample=Image.BICUBIC)

def load_extracted_panels():
    files = glob.glob(os.path.join(EXTRACTED_PANELS_DIR, "*.png"))
    return [(f, Image.open(f).convert("RGBA")) for f in files]

def generate_synthetic_from_real():
    panels_data = load_extracted_panels()
    if not panels_data:
        print("No extracted panels found!")
        return
        
    bg_files = glob.glob(os.path.join(BG_DIR, "*.jpg")) + glob.glob(os.path.join(BG_DIR, "*.png"))
    
    for bg_path in bg_files:
        base_name = os.path.splitext(os.path.basename(bg_path))[0]
        lbl_path = os.path.join(LBL_DIR, f"{base_name}.txt")
        json_path = os.path.join(META_DIR, f"{base_name}_mix.json") # Reference an old JSON for target size/shadow params
        
        if not os.path.exists(lbl_path): continue
        
        # Read placement coordinates from labels
        centers = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) > 1:
                    xs, ys = parts[1::2], parts[2::2]
                    centers.append((sum(xs)/len(xs), sum(ys)/len(ys)))
        
        # Load reference metadata for scale targets (optional, can be hardcoded)
        reference_meta = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                reference_meta = json.load(f)

        bg_img = Image.open(bg_path).convert("RGBA")
        
        # --- VARIANT 1: SAME PANEL ---
        img_same = bg_img.copy()
        chosen_panel_path, chosen_panel_img = random.choice(panels_data)
        new_meta_same = {"variant": "same_panel", "placements": []}
        
        for i, (cx_norm, cy_norm) in enumerate(centers):
            cx, cy = cx_norm * bg_img.width, cy_norm * bg_img.height
            
            # Retrieve locked size from reference JSON, or default to 33
            p_data = reference_meta.get('panels', [{}])
            locked_size = p_data[i % len(p_data)].get('locked_size', 33) 
            
            # Scale panel to match original synthetic constraints
            scale_factor = locked_size / max(chosen_panel_img.size)
            new_w = max(2, int((chosen_panel_img.width * scale_factor) // 2 * 2))
            new_h = max(2, int((chosen_panel_img.height * scale_factor) // 2 * 2))
            scaled_panel = chosen_panel_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Because real panels already have perspective, we apply LESS rotation/stretch
            rot = random.randint(-5, 5) 
            img_geom = apply_geometry(scaled_panel, rot, 1.0, 1.0)
            
            pw, ph = img_geom.size
            px, py = int(cx - pw / 2), int(cy - ph / 2)
            
            img_same.paste(img_geom, (px, py), mask=img_geom)
            
            # Log new metadata
            new_meta_same["placements"].append({
                "source_panel": os.path.basename(chosen_panel_path),
                "cx": round(cx_norm, 4),
                "cy": round(cy_norm, 4),
                "applied_rotation": rot
            })

        img_same.convert("RGB").save(os.path.join(OUTPUT_DIR, f"{base_name}_syn_same.jpg"))
        with open(os.path.join(OUTPUT_META, f"{base_name}_syn_same.json"), "w") as f:
            json.dump(new_meta_same, f, indent=4)
            
            
        # --- VARIANT 2: MIX PANELS ---
        img_mix = bg_img.copy()
        new_meta_mix = {"variant": "mix_panels", "placements": []}
        
        for i, (cx_norm, cy_norm) in enumerate(centers):
            cx, cy = cx_norm * bg_img.width, cy_norm * bg_img.height
            
            # Pick a RANDOM panel for every single placement
            rand_panel_path, rand_panel_img = random.choice(panels_data)
            
            p_data = reference_meta.get('panels', [{}])
            locked_size = p_data[i % len(p_data)].get('locked_size', 33)
            
            scale_factor = locked_size / max(rand_panel_img.size)
            new_w = max(2, int((rand_panel_img.width * scale_factor) // 2 * 2))
            new_h = max(2, int((rand_panel_img.height * scale_factor) // 2 * 2))
            scaled_panel = rand_panel_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            rot = random.randint(-5, 5)
            img_geom = apply_geometry(scaled_panel, rot, 1.0, 1.0)
            
            pw, ph = img_geom.size
            px, py = int(cx - pw / 2), int(cy - ph / 2)
            
            img_mix.paste(img_geom, (px, py), mask=img_geom)
            
            new_meta_mix["placements"].append({
                "source_panel": os.path.basename(rand_panel_path),
                "cx": round(cx_norm, 4),
                "cy": round(cy_norm, 4),
                "applied_rotation": rot
            })

        img_mix.convert("RGB").save(os.path.join(OUTPUT_DIR, f"{base_name}_syn_mix.jpg"))
        with open(os.path.join(OUTPUT_META, f"{base_name}_syn_mix.json"), "w") as f:
            json.dump(new_meta_mix, f, indent=4)

    print("Synthetic generation complete.")