import os
import json
import math
import glob
import random
import numpy as np
import shutil
from PIL import Image, ImageEnhance, ImageFilter

# def load_extracted_panels():
#     files = glob.glob(os.path.join(EXTRACTED_PANELS_DIR, "*.png"))
#     return [(f, Image.open(f).convert("RGBA")) for f in files]


# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# Directories
INPUT_BG_DIR = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26v4'   # Original clean backgrounds
INPUT_LBL_DIR = "dataset_yolo_seg35/labels"       # Labels dictating locations & variant type
INPUT_META_DIR = "dataset_yolo_seg35/meta"        # JSON dictating grid & augmentations
INPUT_PANEL_DIR = "./extracted_panels"            # Folder containing extracted real panels
OUTPUT_DIR = "dataset_synthetic_rebuilt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
PL0_IDENTIFIER = "PL0"

# ==============================================================================
# --- CORE IMAGE MATH & PROCESSING ---
# ==============================================================================

def generate_compact_grid(num_cells): # TODO 2-col only
    """Generates a tightly packed grid layout based only on the number of cells."""
    if num_cells <= 1: return [(0, 0)]
    cells = []
    # Builds a compact 2-column layout to best fit standard bounding boxes
    for i in range(num_cells):
        col = (i % 2) * 2
        row = (i // 2) * 2
        cells.append((col, row))
    return cells

def create_grid_composite_multi(panel_imgs, grid_shape):
    w, h = panel_imgs[0].size
    half_w, half_h = w // 2, h // 2
    
    min_x, max_x = min(g[0] for g in grid_shape), max(g[0] for g in grid_shape)
    min_y, max_y = min(g[1] for g in grid_shape), max(g[1] for g in grid_shape)
    
    gw = (max_x - min_x + 2) * half_w
    gh = (max_y - min_y + 2) * half_h
    
    composite = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
    for i, (gx, gy) in enumerate(grid_shape):
        panel = panel_imgs[i] if i < len(panel_imgs) else panel_imgs[-1]
        px, py = (gx - min_x) * half_w, (gy - min_y) * half_h
        composite.paste(panel, (px, py), mask=panel)
        
    return composite

def scale_panel(panel_img, target_size):
    """Dynamically scales any extracted panel to the JSON's locked_size."""
    scale_factor = target_size / max(panel_img.size)
    new_w = max(2, int((panel_img.width * scale_factor) // 2 * 2))
    new_h = max(2, int((panel_img.height * scale_factor) // 2 * 2))
    return panel_img.resize((new_w, new_h), Image.Resampling.LANCZOS), new_w, new_h

def apply_geometry(img, rot, stretch_x, stretch_y):
    res = img.copy()
    if stretch_x != 1.0 or stretch_y != 1.0:
        new_w, new_h = max(1, int(res.width * stretch_x)), max(1, int(res.height * stretch_y))
        res = res.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return res.rotate(rot, expand=True, resample=Image.BICUBIC)

def apply_noise_np(img, intensity):
    if intensity <= 0: return img
    arr = np.array(img).astype('float32')
    if arr.shape[2] == 4:
        noise = np.random.normal(0, intensity, arr[:,:,:3].shape)
        arr[:,:,:3] = np.clip(arr[:,:,:3] + noise, 0, 255)
    else:
        noise = np.random.normal(0, intensity, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype('uint8'), mode=img.mode)

def apply_photometry(img, bright, noise, blur_radius):
    res = img.copy()
    if bright != 1.0: res = ImageEnhance.Brightness(res).enhance(bright)
    if blur_radius > 0: res = res.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if noise > 0: res = apply_noise_np(res, noise)
    return res

def create_shadow(img, blur_radius, opacity):
    if opacity <= 0 or blur_radius <= 0:
        return Image.new("RGBA", img.size, (0,0,0,0)), 0
        
    pad = int(math.ceil(blur_radius)) * 2 + 2
    shadow = Image.new("RGBA", (img.width + pad * 2, img.height + pad * 2), (0, 0, 0, 0))
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    black.putalpha(img.split()[3].point(lambda p: int(p * opacity)))
    shadow.paste(black, (pad, pad))
    return shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius)), pad

def extract_base_bg_name(label_filename):
    name = label_filename
    for suffix in ["_mix_aug", "_mix", "_comp_aug", "_comp", "_pnl0_aug", "_pnl0", "_aug"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name

# ==============================================================================
# --- MAIN PIPELINE ---
# ==============================================================================

def rebuild_dataset():
    panel_files = glob.glob(os.path.join(INPUT_PANEL_DIR, "*.png"))
    if not panel_files:
        print("No extracted panels found!")
        return

    all_panels = [(f, Image.open(f).convert("RGBA")) for f in panel_files]
    pl0_panel_data = next((p for p in all_panels if PL0_IDENTIFIER in os.path.basename(p[0])), None)
    if not pl0_panel_data:
        pl0_panel_data = random.choice(all_panels)

    lbl_files = glob.glob(os.path.join(INPUT_LBL_DIR, "*.txt"))
    print(f"Found {len(lbl_files)} labels. Rebuilding dataset...\n")

    for lbl_path in lbl_files:
        base_label_name = os.path.splitext(os.path.basename(lbl_path))[0]
        name_lower = base_label_name.lower()
        
        # 1. Determine Variant Context
        if "mix" in name_lower: variant_type = "mix"
        elif "comp" in name_lower: variant_type = "comp"
        elif "pnl0" in name_lower: variant_type = "pnl0"
        elif "empty" in name_lower:
            shutil.copy2(lbl_path, os.path.join(OUTPUT_DIR, f"{base_label_name}.txt"))
            continue
        else: continue

        # 2. Resolve Background
        bg_name = extract_base_bg_name(base_label_name)
        bg_path_jpg = os.path.join(INPUT_BG_DIR, f"{bg_name}.jpg")
        bg_path_png = os.path.join(INPUT_BG_DIR, f"{bg_name}.png")
        
        if os.path.exists(bg_path_jpg): bg_img = Image.open(bg_path_jpg).convert("RGBA")
        elif os.path.exists(bg_path_png): bg_img = Image.open(bg_path_png).convert("RGBA")
        else: continue

        # 3. Load Blueprint Metadata
        json_path = os.path.join(INPUT_META_DIR, f"{base_label_name}.json")
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: meta = json.load(f)
            
        p_data_list = meta.get('panels', [])
        g_meta = meta.get('global_augmentations', {})

        # 4. Extract Polygon Centers from Label File
        centers = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) > 1:
                    xs, ys = parts[1::2], parts[2::2]
                    centers.append((sum(xs)/len(xs), sum(ys)/len(ys)))

        out_img = bg_img.copy()

        # 5. Process Each Placement
        for i, (cx_norm, cy_norm) in enumerate(centers):
            cx, cy = cx_norm * bg_img.width, cy_norm * bg_img.height
            p_meta = p_data_list[i % len(p_data_list)]
            
            # Reconstruct grid shape from cell count
            grid_cells = p_meta.get('grid_cells', 1)
            grid_shape = generate_compact_grid(grid_cells)
            
            locked_size = p_meta.get('locked_size', 33)
            sh_meta = p_meta.get('shadow', {'offset_x':0, 'offset_y':0, 'blur_radius':0, 'opacity':0})

            panel_list = []
            
            if variant_type == "mix":
                _, rand_panel = random.choice(all_panels)
                scaled_panel, _, _ = scale_panel(rand_panel, locked_size)
                panel_list = [scaled_panel] * len(grid_shape)
                
            elif variant_type == "comp":
                cw, ch = 0, 0
                for _ in grid_shape:
                    _, rand_cell_panel = random.choice(all_panels)
                    scaled_cell, temp_w, temp_h = scale_panel(rand_cell_panel, locked_size)
                    if cw == 0: cw, ch = temp_w, temp_h
                    panel_list.append(scaled_cell.resize((cw, ch), Image.Resampling.LANCZOS))
                    
            elif variant_type == "pnl0":
                scaled_pl0, _, _ = scale_panel(pl0_panel_data[1], locked_size)
                panel_list = [scaled_pl0] * len(grid_shape)

            # Render logic
            comp = create_grid_composite_multi(panel_list, grid_shape)
            img_geom = apply_geometry(comp, p_meta.get('rotation_deg', 0), p_meta.get('stretch_x', 1.0), p_meta.get('stretch_y', 1.0))
            img_rot = apply_photometry(img_geom, p_meta.get('brightness_match_multiplier', 1.0), p_meta.get('noise_intensity', 0), p_meta.get('blur_radius', 0))
            
            shadow_img, pad = create_shadow(img_rot, blur_radius=sh_meta['blur_radius'], opacity=sh_meta['opacity'])
            
            pw, ph = img_geom.size
            px, py = int(cx - pw / 2), int(cy - ph / 2)
            shadow_px = px + sh_meta['offset_x'] - pad
            shadow_py = py + sh_meta['offset_y'] - pad
            
            out_img.paste(shadow_img, (int(shadow_px), int(shadow_py)), mask=shadow_img)
            out_img.paste(img_rot, (px, py), mask=img_rot)

        # 6. Apply Global Augmentations
        if 'noise' in g_meta and g_meta['noise'] > 0:
            out_img = apply_noise_np(out_img, g_meta['noise'])
            
        if 'rotation_deg' in g_meta and g_meta['rotation_deg'] != 0:
            out_img = out_img.rotate(g_meta['rotation_deg'], resample=Image.BICUBIC, expand=False)

        # 7. Save Image & Copy original Label
        out_img.convert("RGB").save(os.path.join(OUTPUT_DIR, f"{base_label_name}.jpg"), quality=95)
        shutil.copy2(lbl_path, os.path.join(OUTPUT_DIR, f"{base_label_name}.txt"))
        
        print(f"Reconstructed: {base_label_name}.jpg")

    print("\nDataset generation completed successfully.")

if __name__ == "__main__":
    rebuild_dataset()
