import os
import json
import math
import glob
import random
import numpy as np
import shutil
from PIL import Image, ImageEnhance, ImageFilter

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# Directories
INPUT_BG_DIR = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26v4'   # Original clean backgrounds
INPUT_LBL_DIR = "dataset_yolo_seg35/labels"       # Labels dictating locations & variant type
INPUT_META_DIR = "dataset_yolo_seg35/meta"        # JSON dictating grid & augmentations
INPUT_PANEL_DIR = "./extracted_panels2"            # Folder containing extracted real panels
OUTPUT_DIR = "dataset_synthetic_rebuilt2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
PL0_IDENTIFIER = "pl0"
GRID_SHIFT_PROBABILITY = 0.25
MAX_GRID_RETRIES = 100

# ==============================================================================
# --- CORE GEOMETRY & MATH ---
# ==============================================================================

def overlaps(c1, c2):
    return abs(c1[0] - c2[0]) < 2 and abs(c1[1] - c2[1]) < 2

def generate_random_grid(num_cells):
    if num_cells <= 1: return [(0, 0)]
    cells = [(0, 0)]
    if random.random() < GRID_SHIFT_PROBABILITY:
        valid_moves = [(0, -2), (-1, -2), (1, -2), (0, 2), (-1, 2), (1, 2), (-2, 0), (-2, -1), (-2, 1), (2, 0), (2, -1), (2, 1)]
    else:
        valid_moves = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    adj = set(valid_moves)
    for _ in range(num_cells - 1):
        valid_adj = [m for m in adj if not any(overlaps(m, c) for c in cells)]
        if not valid_adj: break
        new_cell = random.choice(valid_adj)
        cells.append(new_cell)
        if new_cell in adj: adj.remove(new_cell)
        for dx, dy in valid_moves:
            neighbor = (new_cell[0] + dx, new_cell[1] + dy)
            if not any(overlaps(neighbor, c) for c in cells): adj.add(neighbor)
    return cells

def generate_compact_grid(num_cells):
    if num_cells <= 1: return [(0, 0)]
    return [((i % 2) * 2, (i // 2) * 2) for i in range(num_cells)]

def transform_polygon(polygon, gw, gh, cx, cy, rot_deg, stretch_x, stretch_y):
    """Mathematically maps points relative to local center, then moves to global center."""
    angle_rad = math.radians(-rot_deg)
    transformed = []
    for x, y in polygon:
        # 1. Shift to center of the composite
        x_c, y_c = x - gw / 2.0, y - gh / 2.0
        # 2. Stretch
        sx, sy = x_c * stretch_x, y_c * stretch_y
        # 3. Rotate
        rx = sx * math.cos(angle_rad) - sy * math.sin(angle_rad)
        ry = sx * math.sin(angle_rad) + sy * math.cos(angle_rad)
        # 4. Translate to final placement center
        transformed.append((cx + rx, cy + ry))
    return transformed

def rotate_point(x, y, cx, cy, angle_rad):
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + cx, ry + cy

def recover_best_grid_shape(grid_cells, cell_w, cell_h, rot, stretch_x, stretch_y, target_w, target_h):
    if target_w == 0 or target_h == 0: return generate_compact_grid(grid_cells)
    best_grid = None
    best_error = float('inf')

    for _ in range(MAX_GRID_RETRIES):
        candidate = generate_random_grid(grid_cells)
        min_x, max_x = min(g[0] for g in candidate), max(g[0] for g in candidate)
        min_y, max_y = min(g[1] for g in candidate), max(g[1] for g in candidate)
        
        gw = (max_x - min_x + 2) * (cell_w / 2.0)
        gh = (max_y - min_y + 2) * (cell_h / 2.0)
        
        # Test transform using a pure bounding box
        base_box = [(0,0), (gw,0), (gw,gh), (0,gh)]
        trans_box = transform_polygon(base_box, gw, gh, 0, 0, rot, stretch_x, stretch_y)
        
        cand_w = max(x for x, y in trans_box) - min(x for x, y in trans_box)
        cand_h = max(y for x, y in trans_box) - min(y for x, y in trans_box)
        
        error = abs(cand_w - target_w) / target_w + abs(cand_h - target_h) / target_h
        if error < best_error:
            best_error = error
            best_grid = candidate
        if error < 0.05: break
            
    return best_grid if best_grid else generate_compact_grid(grid_cells)

# ==============================================================================
# --- IMAGE COMPOSITING ---
# ==============================================================================

def scale_panel(panel_img, target_size):
    scale_factor = target_size / max(panel_img.size)
    new_w = max(2, int((panel_img.width * scale_factor) // 2 * 2))
    new_h = max(2, int((panel_img.height * scale_factor) // 2 * 2))
    return panel_img.resize((new_w, new_h), Image.Resampling.LANCZOS), new_w, new_h

def create_grid_composite_and_polys(panel_data_list, grid_shape):
    """Builds the visual image AND maps the complex polygons into pixel coordinates."""
    w, h = panel_data_list[0][0].size
    half_w, half_h = w / 2.0, h / 2.0
    
    min_x, max_x = min(g[0] for g in grid_shape), max(g[0] for g in grid_shape)
    min_y, max_y = min(g[1] for g in grid_shape), max(g[1] for g in grid_shape)
    
    gw = int((max_x - min_x + 2) * half_w)
    gh = int((max_y - min_y + 2) * half_h)
    
    composite = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
    composite_polygons = []
    
    for i, (gx, gy) in enumerate(grid_shape):
        panel_img, panel_poly = panel_data_list[i] if i < len(panel_data_list) else panel_data_list[-1]
        px = int((gx - min_x) * half_w)
        py = int((gy - min_y) * half_h)
        composite.paste(panel_img, (px, py), mask=panel_img)
        
        cw, ch = panel_img.size
        if panel_poly:
            # Map normalized poly coordinates to pixel coordinates in the composite
            cell_poly = [(px + nx * cw, py + ny * ch) for nx, ny in panel_poly]
            composite_polygons.append(cell_poly)
            
    return composite, composite_polygons, gw, gh

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
    if opacity <= 0 or blur_radius <= 0: return Image.new("RGBA", img.size, (0,0,0,0)), 0
    pad = int(math.ceil(blur_radius)) * 2 + 2
    shadow = Image.new("RGBA", (img.width + pad * 2, img.height + pad * 2), (0, 0, 0, 0))
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    black.putalpha(img.split()[3].point(lambda p: int(p * opacity)))
    shadow.paste(black, (pad, pad))
    return shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius)), pad

def extract_base_bg_name(label_filename):
    name = label_filename
    pv_suf = [f'_pnl{p_idx}' for p_idx in range(11)]
    pv_suf_a = [f'_pnl{p_idx}_aug' for p_idx in range(11)]
    pv_suf.extend(pv_suf_a)
    pv_suf.extend(["_mix_aug", "_mix", "_composite_aug", "_composite"])
    for suffix in pv_suf:
        if name.endswith(suffix): return name[:-len(suffix)]
    return name

# ==============================================================================
# --- MAIN PIPELINE ---
# ==============================================================================

def rebuild_dataset():
    panel_files = glob.glob(os.path.join(INPUT_PANEL_DIR, "*.png"))
    if not panel_files:
        print("No extracted panels found!")
        return

    # Load images AND their extracted geometries
    all_panels = []
    for f in panel_files:
        img = Image.open(f).convert("RGBA")
        json_path = f.replace(".png", ".json")
        poly = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as jf:
                poly = json.load(jf).get("polygon", [])
        all_panels.append((f, img, poly))
        
    pl0_panel_data = next((p for p in all_panels if PL0_IDENTIFIER in os.path.basename(p[0])), None)
    if not pl0_panel_data: pl0_panel_data = random.choice(all_panels)

    lbl_files = glob.glob(os.path.join(INPUT_LBL_DIR, "*.txt"))
    print(f"Found {len(lbl_files)} labels. Rebuilding dataset...\n")

    for lbl_path in [lbl_files[0]]:
        print(lbl_path)
        base_label_name = os.path.splitext(os.path.basename(lbl_path))[0]
        name_lower = base_label_name
        
        if "mix" in name_lower: variant_type = "mix"
        elif "comp" in name_lower: variant_type = "comp"
        elif "pnl0" in name_lower: variant_type = "pnl0"
        elif "empty" in name_lower:
            shutil.copy2(lbl_path, os.path.join(OUTPUT_DIR, f"{base_label_name}.txt"))
            with open(os.path.join(OUTPUT_DIR, f"{base_label_name}.json"), "w") as f:
                json.dump({"global_augmentations": {}, "panels": []}, f, indent=4)
            continue
        else: continue

        bg_name = extract_base_bg_name(base_label_name)
        bg_path_jpg = os.path.join(INPUT_BG_DIR, f"{bg_name}.jpg")
        bg_path_png = os.path.join(INPUT_BG_DIR, f"{bg_name}.tif")
        
        if os.path.exists(bg_path_jpg): bg_img = Image.open(bg_path_jpg).convert("RGBA")
        elif os.path.exists(bg_path_png): bg_img = Image.open(bg_path_png).convert("RGBA")
        else: continue

        json_path = os.path.join(INPUT_META_DIR, f"{base_label_name}.json")
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: meta = json.load(f)
            
        p_data_list = meta.get('panels', [])
        g_meta = meta.get('global_augmentations', {})

        placements_data = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) > 1:
                    cls_id = int(parts[0])
                    xs, ys = parts[1::2], parts[2::2]
                    cx_norm, cy_norm = sum(xs)/len(xs), sum(ys)/len(ys)
                    w_norm, h_norm = max(xs) - min(xs), max(ys) - min(ys)
                    placements_data.append((cls_id, cx_norm, cy_norm, w_norm, h_norm))

        out_img = bg_img.copy()
        new_meta = {"global_augmentations": g_meta, "panels": []}
        placement_polygons_pixels = []

        for i, (cls_id, cx_norm, cy_norm, w_norm, h_norm) in enumerate(placements_data):
            cx, cy = cx_norm * bg_img.width, cy_norm * bg_img.height
            orig_w, orig_h = w_norm * bg_img.width, h_norm * bg_img.height
            p_meta = p_data_list[i % len(p_data_list)]
            
            grid_cells = p_meta.get('grid_cells', 1)
            locked_size = p_meta.get('locked_size', 33)
            rot = p_meta.get('rotation_deg', 0)
            sx, sy = p_meta.get('stretch_x', 1.0), p_meta.get('stretch_y', 1.0)

            _, test_panel_img, _ = random.choice(all_panels)
            _, cell_w, cell_h = scale_panel(test_panel_img, locked_size)

            best_grid_shape = recover_best_grid_shape(grid_cells, cell_w, cell_h, rot, sx, sy, orig_w, orig_h)

            panel_list = []
            source_panels = []
            
            if variant_type == "mix":
                p_path, p_img, p_poly = random.choice(all_panels)
                scaled_img, _, _ = scale_panel(p_img, locked_size)
                panel_list = [(scaled_img, p_poly)] * len(best_grid_shape)
                source_panels = [os.path.basename(p_path)] * len(best_grid_shape)
                
            elif variant_type == "comp":
                cw, ch = 0, 0
                for _ in best_grid_shape:
                    p_path, p_img, p_poly = random.choice(all_panels)
                    scaled_img, temp_w, temp_h = scale_panel(p_img, locked_size)
                    if cw == 0: cw, ch = temp_w, temp_h
                    panel_list.append((scaled_img.resize((cw, ch), Image.Resampling.LANCZOS), p_poly))
                    source_panels.append(os.path.basename(p_path))
                    
            elif variant_type == "pnl0":
                p_path, p_img, p_poly = pl0_panel_data
                scaled_img, _, _ = scale_panel(p_img, locked_size)
                panel_list = [(scaled_img, p_poly)] * len(best_grid_shape)
                source_panels = [os.path.basename(p_path)] * len(best_grid_shape)

            # RENDER: Pass the list of tuples (img, poly) to get exact coordinate maps
            comp, comp_polys, gw, gh = create_grid_composite_and_polys(panel_list, best_grid_shape)
            
            img_geom = apply_geometry(comp, rot, sx, sy)
            img_rot = apply_photometry(img_geom, p_meta.get('brightness_match_multiplier', 1.0), p_meta.get('noise_intensity', 0), p_meta.get('blur_radius', 0))
            
            sh_meta = p_meta.get('shadow', {'offset_x':0, 'offset_y':0, 'blur_radius':0, 'opacity':0})
            shadow_img, pad = create_shadow(img_rot, blur_radius=sh_meta['blur_radius'], opacity=sh_meta['opacity'])
            
            pw, ph = img_geom.size
            px_paste = int(cx - pw / 2)
            py_paste = int(cy - ph / 2)
            shadow_px = px_paste + sh_meta['offset_x'] - pad
            shadow_py = py_paste + sh_meta['offset_y'] - pad
            
            out_img.paste(shadow_img, (int(shadow_px), int(shadow_py)), mask=shadow_img)
            out_img.paste(img_rot, (px_paste, py_paste), mask=img_rot)

            # --- MAP THE EXACT POLYGONS ---
            for c_poly in comp_polys:
                trans_poly = transform_polygon(c_poly, gw, gh, cx, cy, rot, sx, sy)
                placement_polygons_pixels.append((cls_id, trans_poly))
            
            updated_p_meta = p_meta.copy()
            updated_p_meta["grid_shape"] = best_grid_shape
            updated_p_meta["source_panels"] = source_panels
            new_meta["panels"].append(updated_p_meta)

        global_rot = g_meta.get('rotation_deg', 0)
        if 'noise' in g_meta and g_meta['noise'] > 0: out_img = apply_noise_np(out_img, g_meta['noise'])
        if global_rot != 0: out_img = out_img.rotate(global_rot, resample=Image.BICUBIC, expand=False)

        # Apply Global Rotation to Polygons & Normalize
        final_labels = []
        cx_img, cy_img = out_img.width / 2.0, out_img.height / 2.0
        global_rot_rad = math.radians(-global_rot)

        for c_id, poly in placement_polygons_pixels:
            new_poly = []
            for px, py in poly:
                rx, ry = rotate_point(px, py, cx_img, cy_img, global_rot_rad) if global_rot != 0 else (px, py)
                nx = max(0, min(out_img.width, rx)) / out_img.width
                ny = max(0, min(out_img.height, ry)) / out_img.height
                new_poly.append((nx, ny))
            final_labels.append((c_id, new_poly))

        out_img.convert("RGB").save(os.path.join(OUTPUT_DIR, f"{base_label_name}.jpg"), quality=95)
        
        with open(os.path.join(OUTPUT_DIR, f"{base_label_name}.txt"), "w") as f:
            for c_id, pts in final_labels:
                coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
                f.write(f"{c_id} {coords}\n")
                
        with open(os.path.join(OUTPUT_DIR, f"{base_label_name}.json"), "w") as f:
            json.dump(new_meta, f, indent=4)
        
        print(f"Reconstructed [{variant_type.upper()}]: {base_label_name}.jpg")

    print("\nDataset generation, precise labels, and metadata export completed successfully.")

if __name__ == "__main__":
    rebuild_dataset()
    