import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
import os
import math
import random
import json
import copy
import shutil

# ==============================================================================
# --- CONFIGURATION CONSTANTS ---
# ==============================================================================

# Global Seed for Reproducibility (Set to None for unpredictable randomness)
RANDOM_SEED = 42

# Directories & Files
OUTPUT_BASE_DIR = "dataset_yolo_seg32"
IMG_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")
META_DIR = os.path.join(OUTPUT_BASE_DIR, "meta")
REAL_DIR = os.path.join(OUTPUT_BASE_DIR, "rzeczywiste")
SKIPPED_FILE = os.path.join(OUTPUT_BASE_DIR, "skipped.txt") 

DEFAULT_BG_FOLDER = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26'
DEFAULT_PANEL_FOLDER = './pvs'

# Grid Generation
GRID_CELLS_MIN = 2
GRID_CELLS_MAX = 6
GRID_SHIFT_PROBABILITY = 0.25  # 20% chance for a staggered (half-shift) grid layout

# Panel Base Geometry (Locked per placement click)
PANEL_SIZE_MIN = 22
PANEL_SIZE_MAX = 44

# Per-Panel Variant Augmentations (Randomized individually during batch save)
ROT_MIN = -180
ROT_MAX = 180
STRETCH_MIN = 0.8
STRETCH_MAX = 1.2
NOISE_MIN = 0
NOISE_MAX = 6
BLUR_MIN = 0.0
BLUR_MAX = 1.0

# Global Image Shadow Parameters (Consistent per final variant image)
SHADOW_LENGTH_MIN = 1.0
SHADOW_LENGTH_MAX = 4.0 # 8
SHADOW_BLUR_MIN = 2.0
SHADOW_BLUR_MAX = 4.0

# Global Image Extra Variant Augmentations
GLOBAL_NOISE_MIN = 6
GLOBAL_NOISE_MAX = 12
GLOBAL_ROT_RANGE_NEG = (-15, -5)
GLOBAL_ROT_RANGE_POS = (5, 15)

# Photometry & Math settings
BRIGHTNESS_CLAMP_MIN = 0.2
BRIGHTNESS_CLAMP_MAX = 2.0
ALPHA_MASK_THRESHOLD = 10

# UI Ghost Preview Settings
UI_GHOST_SCALE_BASE = 33

# Apply Seed globally if defined
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# ==============================================================================
# --- APPLICATION LOGIC ---
# ==============================================================================

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

def overlaps(c1, c2):
    return abs(c1[0] - c2[0]) < 2 and abs(c1[1] - c2[1]) < 2

def generate_random_grid(num_cells):
    if num_cells <= 1: return [(0, 0)]
    cells = [(0, 0)]
    
    allow_shifts = random.random() < GRID_SHIFT_PROBABILITY
    
    if allow_shifts:
        valid_moves = [
            (0, -2), (-1, -2), (1, -2), 
            (0, 2), (-1, 2), (1, 2),    
            (-2, 0), (-2, -1), (-2, 1), 
            (2, 0), (2, -1), (2, 1)     
        ]
    else:
        valid_moves = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    
    adj = set(valid_moves)
    
    for _ in range(num_cells - 1):
        valid_adj = []
        for move in adj:
            if not any(overlaps(move, c) for c in cells):
                valid_adj.append(move)
        
        if not valid_adj: break
        
        new_cell = random.choice(valid_adj)
        cells.append(new_cell)
        if new_cell in adj:
            adj.remove(new_cell)
        
        for dx, dy in valid_moves:
            neighbor = (new_cell[0] + dx, new_cell[1] + dy)
            if not any(overlaps(neighbor, c) for c in cells):
                adj.add(neighbor)
                
    return cells

def create_grid_composite(panel_img, grid_shape):
    w, h = panel_img.size
    half_w = w // 2
    half_h = h // 2
    
    min_x = min(gx for gx, gy in grid_shape)
    max_x = max(gx for gx, gy in grid_shape)
    min_y = min(gy for gx, gy in grid_shape)
    max_y = max(gy for gx, gy in grid_shape)
    
    gw = (max_x - min_x + 2) * half_w
    gh = (max_y - min_y + 2) * half_h
    
    composite = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
    
    for gx, gy in grid_shape:
        px = (gx - min_x) * half_w
        py = (gy - min_y) * half_h
        composite.paste(panel_img, (px, py), mask=panel_img)
        
    return composite, w, h

def get_grid_polygon(grid_shape, cell_w, cell_h):
    half_w = cell_w / 2.0
    half_h = cell_h / 2.0
    
    min_x = min(gx for gx, gy in grid_shape)
    min_y = min(gy for gx, gy in grid_shape)
    max_x = max(gx for gx, gy in grid_shape)
    max_y = max(gy for gx, gy in grid_shape)
    
    edges = set()
    for gx, gy in grid_shape:
        cx = gx - min_x
        cy = gy - min_y
        
        cell_edges = [
            ((cx, cy), (cx+1, cy)),       
            ((cx+1, cy), (cx+2, cy)),     
            ((cx+2, cy), (cx+2, cy+1)),   
            ((cx+2, cy+1), (cx+2, cy+2)), 
            ((cx+2, cy+2), (cx+1, cy+2)), 
            ((cx+1, cy+2), (cx, cy+2)),   
            ((cx, cy+2), (cx, cy+1)),     
            ((cx, cy+1), (cx, cy))        
        ]
        
        for edge in cell_edges:
            reverse_edge = (edge[1], edge[0])
            if reverse_edge in edges:
                edges.remove(reverse_edge) 
            else:
                edges.add(edge)
                
    adj = {start: end for start, end in edges}
    
    start_node = next(iter(adj.keys()))
    polygon = []
    current_node = start_node
    
    while True:
        polygon.append(current_node)
        current_node = adj[current_node]
        if current_node == start_node:
            break
            
    comp_w_cells = (max_x - min_x + 2)
    comp_h_cells = (max_y - min_y + 2)
    center_x = comp_w_cells / 2.0
    center_y = comp_h_cells / 2.0
    
    centered_poly = []
    for nx, ny in polygon:
        px = (nx - center_x) * half_w
        py = (ny - center_y) * half_h
        centered_poly.append((px, py))
        
    return centered_poly

def transform_polygon(polygon, cx, cy, rot_deg, stretch_x, stretch_y):
    angle_rad = math.radians(-rot_deg)
    transformed = []
    for x, y in polygon:
        sx = x * stretch_x
        sy = y * stretch_y
        
        rx = sx * math.cos(angle_rad) - sy * math.sin(angle_rad)
        ry = sx * math.sin(angle_rad) + sy * math.cos(angle_rad)
        
        transformed.append((cx + rx, cy + ry))
    return transformed

def normalize_polygon(polygon, img_w, img_h):
    norm_poly = []
    for x, y in polygon:
        nx = max(0, min(img_w, x)) / img_w 
        ny = max(0, min(img_h, y)) / img_h 
        norm_poly.append((nx, ny))
    return norm_poly

def apply_geometry(img, rot, stretch_x, stretch_y):
    res = img.copy()
    if stretch_x != 1.0 or stretch_y != 1.0:
        new_w = max(1, int(res.width * stretch_x))
        new_h = max(1, int(res.height * stretch_y))
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
    if bright != 1.0: 
        res = ImageEnhance.Brightness(res).enhance(bright)
    if blur_radius > 0: 
        res = res.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if noise > 0: 
        res = apply_noise_np(res, noise)
    return res

def calculate_local_shadow_opacity(bg_img, px, py, pw, ph):
    """Calculates shadow intensity based on local background contrast (standard deviation)."""
    px_safe = max(0, px)
    py_safe = max(0, py)
    pw_safe = min(bg_img.width, px + pw)
    ph_safe = min(bg_img.height, py + ph)
    
    bg_crop = bg_img.crop((px_safe, py_safe, pw_safe, ph_safe))
    bg_arr = np.array(bg_crop.convert("L"), dtype=np.float32)
    
    if bg_arr.size == 0:
        return 0.35
        
    std = np.std(bg_arr) / 255.0
    intensity = 0.15 + 0.4 * std
    return float(max(0.1, min(intensity, 0.7)))

def create_shadow(img, blur_radius, opacity):
    """Generates a pure PIL shadow based on the rotated panel silhouette."""
    pad = int(math.ceil(blur_radius)) * 2 + 2
    padded_size = (img.width + pad * 2, img.height + pad * 2)
    shadow = Image.new("RGBA", padded_size, (0, 0, 0, 0))
    
    alpha = img.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity))
    
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    black.putalpha(alpha)
    
    shadow.paste(black, (pad, pad))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return shadow, pad

def calculate_brightness_adjustment(bg_img, panel_geom_img, px, py):
    pw, ph = panel_geom_img.size
    bg_crop = bg_img.crop((px, py, px + pw, py + ph))
    
    bg_arr = np.array(bg_crop.convert("RGBA"), dtype=np.float32)
    panel_arr = np.array(panel_geom_img.convert("RGBA"), dtype=np.float32)
    
    alpha = panel_arr[:, :, 3]
    mask = alpha > ALPHA_MASK_THRESHOLD 
    
    if not np.any(mask): return 1.0
        
    def get_lum(arr):
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        
    bg_lum = np.mean(get_lum(bg_arr)[mask])
    panel_lum = np.mean(get_lum(panel_arr)[mask])
    
    if panel_lum < 1.0: panel_lum = 1.0 
    
    factor = float(bg_lum) / float(panel_lum)
    return float(max(BRIGHTNESS_CLAMP_MIN, min(factor, BRIGHTNESS_CLAMP_MAX)))

def rotate_point(x, y, cx, cy, angle_rad):
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + cx, ry + cy

def generate_dataset_variants(original_img, original_labels_poly, base_panel_augs, base_name):
    w_img, h_img = original_img.size
    
    img_aug = original_img.copy()
    labels_aug = [(cls, list(pts)) for cls, pts in original_labels_poly]

    global_noise_val = random.randint(GLOBAL_NOISE_MIN, GLOBAL_NOISE_MAX)
    img_aug = apply_noise_np(img_aug, global_noise_val)

    angle = random.choice([
        random.randint(GLOBAL_ROT_RANGE_NEG[0], GLOBAL_ROT_RANGE_NEG[1]), 
        random.randint(GLOBAL_ROT_RANGE_POS[0], GLOBAL_ROT_RANGE_POS[1])
    ])
    
    img_aug = img_aug.rotate(angle, resample=Image.BICUBIC, expand=False)
    angle_rad = math.radians(-angle)
    cx_img, cy_img = w_img / 2, h_img / 2
    
    new_labels = []
    for cls, points in labels_aug:
        new_pts = []
        for nx, ny in points:
            px, py = nx * w_img, ny * h_img
            rx, ry = rotate_point(px, py, cx_img, cy_img, angle_rad)
            rx = max(0, min(w_img, rx))
            ry = max(0, min(h_img, ry))
            new_pts.append((rx / w_img, ry / h_img))
        new_labels.append((cls, new_pts))
    labels_aug = new_labels

    f_name = f"{base_name}_aug"
    img_aug.convert("RGB").save(os.path.join(IMG_DIR, f"{f_name}.jpg"), quality=95)
    
    with open(os.path.join(LBL_DIR, f"{f_name}.txt"), "w") as f:
        for cls, pts in labels_aug:
            coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
            f.write(f"{cls} {coords}\n")
            
    aug_data = {
        "global_augmentations": {
            "noise": global_noise_val,
            "rotation_deg": angle
        },
        "panels": copy.deepcopy(base_panel_augs)
    }
    with open(os.path.join(META_DIR, f"{f_name}.json"), "w") as f:
        json.dump(aug_data, f, indent=4)

# ==============================================================================
# --- GUI APPLICATION ---
# ==============================================================================

class YoloObbApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Seg Generator (Variant-Locked Shadows)")
        self.root.geometry("1280x800")

        self.bg_images = []
        self.panel_images = []
        self.current_bg_idx = 0
        
        self.base_img = None
        self.work_img = None
        self.preview_panel_img = None
        
        self.placements = [] 
        self.tk_preview = None
        self.tk_shadow = None
        
        self.last_x = 0
        self.last_y = 0
        
        self.saved_empty_bgs = set()
        
        # For UI preview purposes only
        self.ui_shadow_dx = 2.0
        self.ui_shadow_dy = 2.0
        self.ui_shadow_blur = 3.0
        
        self.roll_new_geometry()

        self.setup_ui()

    def setup_ui(self):
        ctrl = tk.Frame(self.root, width=300, bg="#dddddd", padx=10, pady=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(ctrl, text="BATCH GENERATOR", font=("Arial", 14, "bold"), bg="#dddddd").pack(pady=10)
        
        tk.Button(ctrl, text="1. Wczytaj folder TŁA", command=self.load_bg_folder, bg="white").pack(fill=tk.X, pady=5)
        tk.Button(ctrl, text="2. Wczytaj folder PANELI", command=self.load_panel_folder, bg="white").pack(fill=tk.X, pady=5)

        tk.Label(ctrl, text="--- Nawigacja Tła ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        self.lbl_bg_info = tk.Label(ctrl, text="Tło: 0/0", bg="#dddddd")
        self.lbl_bg_info.pack()
        
        tk.Button(ctrl, text="Pomiń Tło (Trwale)", command=self.skip_bg, bg="#ff9999").pack(fill=tk.X, pady=5)
        tk.Button(ctrl, text="Oznacz jako 'Rzeczywiste' i pomiń", command=self.mark_as_real, bg="#64b5f6", fg="white", font=("Arial", 9, "bold")).pack(fill=tk.X, pady=5)

        tk.Label(ctrl, text="--- Parametry Generacji ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        
        info_font = ("Arial", 9, "bold")
        info_color = "#2e7d32"
        tk.Label(ctrl, text=f"Rozmiar: LOSOWY ({PANEL_SIZE_MIN}-{PANEL_SIZE_MAX}px) ZABLOK.", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text=f"Anizotropia: LOSOWA ({STRETCH_MIN}-{STRETCH_MAX}x)", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text=f"Rozmycie: LOSOWE ({BLUR_MIN}-{BLUR_MAX}px)", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text=f"Obrót: LOSOWY ({ROT_MIN}° do {ROT_MAX}°)", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        
        tk.Label(ctrl, text="Cień: SPÓJNY KIERUNEK (PER OBRAZ)", bg="#dddddd", fg="#b71c1c", font=("Arial", 9, "bold", "italic")).pack(anchor="w", pady=(8,2))
        tk.Label(ctrl, text="Jasność: DOPASOWANA DO TŁA", bg="#dddddd", fg="#b71c1c", font=("Arial", 9, "bold", "italic")).pack(anchor="w", pady=2)
        
        tk.Label(ctrl, text="--- Sterowanie ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        tk.Label(ctrl, text="LEWY KLIK: Postaw Grid z podglądu", bg="#dddddd", fg="#b71c1c", font=("Arial", 9, "bold")).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="PRAWY KLIK: Postaw Pojedynczy Panel", bg="#dddddd", fg="#0d47a1", font=("Arial", 9, "bold")).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="SPACJA: Zmień podgląd na inny układ", bg="#dddddd", fg="#d84315", font=("Arial", 9, "bold")).pack(anchor="w", pady=2)

        tk.Label(ctrl, text="--- Zapis ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        self.btn_save = tk.Button(ctrl, text="ZAPISZ WSZYSTKIE WARIANTY", command=self.save_batch, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=10, ipady=5)
        
        tk.Button(ctrl, text="Wyczyść kliknięcia", command=self.reset_canvas).pack(fill=tk.X)
        
        self.lbl_status = tk.Label(ctrl, text="Gotowy", bg="#dddddd", fg="blue")
        self.lbl_status.pack(side=tk.BOTTOM, pady=10)

        self.cv_frame = tk.Frame(self.root, bg="#333")
        self.cv_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(self.cv_frame, bg="#333", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", lambda e: self.on_click(e, is_grid=True))
        self.canvas.bind("<Button-2>", lambda e: self.on_click(e, is_grid=False)) 
        self.canvas.bind("<Button-3>", lambda e: self.on_click(e, is_grid=False))
        self.canvas.bind("<Motion>", self.on_move)
        
        self.root.bind("<space>", self.on_space)

    def load_bg_folder(self):
        folder = DEFAULT_BG_FOLDER
        if folder:
            all_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            existing_labels = os.listdir(LBL_DIR) if os.path.exists(LBL_DIR) else []
            existing_reals = os.listdir(REAL_DIR) if os.path.exists(REAL_DIR) else []
            
            skipped_bgs = set()
            if os.path.exists(SKIPPED_FILE):
                with open(SKIPPED_FILE, "r") as f_skip:
                    skipped_bgs = set(line.strip() for line in f_skip if line.strip())
            
            unprocessed = []
            for f in all_files:
                base_name = os.path.splitext(f)[0]
                prefix = f"{base_name}_"
                
                has_labels = any(lbl.startswith(prefix) for lbl in existing_labels)
                is_real = f in existing_reals
                is_skipped = f in skipped_bgs
                
                if not has_labels and not is_real and not is_skipped:
                    unprocessed.append(os.path.join(folder, f))
            
            self.bg_images = unprocessed
            
            if self.bg_images:
                self.current_bg_idx = 0
                self.load_current_bg()
                self.lbl_status.config(text=f"Wczytano {len(self.bg_images)} nowych tła (Pominięto przetworzone/wykluczone).")
            else:
                self.lbl_bg_info.config(text="Tło: 0/0")
                messagebox.showinfo("Gotowe", "Wszystkie tła w tym folderze zostały już przetworzone, skopiowane lub trwale pominięte!")

    def load_panel_folder(self):
        folder = DEFAULT_PANEL_FOLDER
        if folder:
            self.panel_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.panel_images:
                self.preview_panel_img = Image.open(self.panel_images[0]).convert("RGBA")
                self.roll_new_geometry()
                self.lbl_status.config(text=f"Wczytano {len(self.panel_images)} paneli")
                self.check_ready_state()
            else:
                messagebox.showwarning("Pusto", "Brak obrazów w folderze paneli.")

    def load_current_bg(self):
        if not self.bg_images: return
        p = self.bg_images[self.current_bg_idx]
        self.base_img = Image.open(p).convert("RGBA")
        self.lbl_bg_info.config(text=f"Tło: {self.current_bg_idx + 1} / {len(self.bg_images)}")
        
        # Roll a temporary shadow direction just for the ghost UI preview
        ui_angle = random.uniform(0, 2 * math.pi)
        ui_len = random.uniform(SHADOW_LENGTH_MIN, SHADOW_LENGTH_MAX)
        self.ui_shadow_dx = math.cos(ui_angle) * ui_len
        self.ui_shadow_dy = math.sin(ui_angle) * ui_len
        self.ui_shadow_blur = random.uniform(SHADOW_BLUR_MIN, SHADOW_BLUR_MAX)
        
        self.reset_canvas()
        self.check_ready_state()

    def next_bg(self):
        if not self.bg_images: return
        self.current_bg_idx = (self.current_bg_idx + 1) % len(self.bg_images)
        self.load_current_bg()
        
    def skip_bg(self):
        if not self.bg_images: return
        
        current_bg_path = self.bg_images[self.current_bg_idx]
        filename = os.path.basename(current_bg_path)
        
        try:
            with open(SKIPPED_FILE, "a") as f_skip:
                f_skip.write(filename + "\n")
        except Exception as e:
            print(f"Failed to write to skipped file: {e}")

        self.lbl_status.config(text=f"Pominięto i trwale wykluczono: {filename}")
        
        self.bg_images.pop(self.current_bg_idx)
        
        if self.bg_images:
            self.current_bg_idx = self.current_bg_idx % len(self.bg_images)
            self.load_current_bg()
        else:
            self.canvas.delete("all")
            self.lbl_bg_info.config(text="Tło: 0/0")
            messagebox.showinfo("Koniec", "Nie ma więcej teł w kolejce.")
        
    def mark_as_real(self):
        if not self.bg_images: return
        current_bg_path = self.bg_images[self.current_bg_idx]
        filename = os.path.basename(current_bg_path)
        dest_path = os.path.join(REAL_DIR, filename)
        
        try:
            shutil.copy2(current_bg_path, dest_path)
            self.lbl_status.config(text=f"Skopiowano do 'rzeczywiste': {filename}")
            
            self.bg_images.pop(self.current_bg_idx)
            
            if self.bg_images:
                self.current_bg_idx = self.current_bg_idx % len(self.bg_images)
                self.root.after(500, self.load_current_bg) 
            else:
                self.canvas.delete("all")
                self.lbl_bg_info.config(text="Tło: 0/0")
                messagebox.showinfo("Koniec", "Nie ma więcej teł w kolejce.")
                
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się skopiować: {str(e)}")

    def check_ready_state(self):
        if self.base_img and self.panel_images:
            self.btn_save.config(state=tk.NORMAL)

    def reset_canvas(self):
        if self.base_img:
            self.work_img = self.base_img.copy()
            self.placements = []
            self.redraw()
            self.draw_ghost()
            
    def roll_new_geometry(self):
        self.current_grid_shape = generate_random_grid(random.randint(GRID_CELLS_MIN, GRID_CELLS_MAX))
        self.current_size = random.randint(PANEL_SIZE_MIN, PANEL_SIZE_MAX)

    def _get_scaled_preview_panel(self):
        scale_factor = self.current_size / max(self.preview_panel_img.size)
        scaled_w = max(2, int(self.preview_panel_img.width * scale_factor))
        scaled_w = (scaled_w // 2) * 2 
        scaled_h = max(2, int(self.preview_panel_img.height * scale_factor))
        scaled_h = (scaled_h // 2) * 2 
        return self.preview_panel_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    def draw_ghost(self):
        if not self.work_img or not self.preview_panel_img: return
        
        self.canvas.delete("ghost")
        
        scaled_panel = self._get_scaled_preview_panel()
        comp, _, _ = create_grid_composite(scaled_panel, self.current_grid_shape)
        
        img_geom = apply_geometry(comp, 0, 1.0, 1.0)
        
        pw, ph = img_geom.size
        px, py = int(self.last_x - pw / 2), int(self.last_y - ph / 2)
        auto_bright = calculate_brightness_adjustment(self.base_img, img_geom, px, py)
        
        processed_ov = apply_photometry(img_geom, auto_bright, 0, 0.0)
        
        # Hardcoded opacity and extended length to make UI preview highly visible
        ghost_opacity = 0.9 
        ghost_length = max(4.0, math.hypot(self.ui_shadow_dx, self.ui_shadow_dy) * 1.5)
        
        shadow_img, pad = create_shadow(processed_ov, blur_radius=self.ui_shadow_blur, opacity=ghost_opacity)
        
        self.tk_shadow = ImageTk.PhotoImage(shadow_img)
        self.tk_preview = ImageTk.PhotoImage(processed_ov)
        
        # Normalize the UI direction and apply the visible length
        ui_magnitude = math.hypot(self.ui_shadow_dx, self.ui_shadow_dy)
        if ui_magnitude > 0:
            norm_dx = (self.ui_shadow_dx / ui_magnitude) * ghost_length
            norm_dy = (self.ui_shadow_dy / ui_magnitude) * ghost_length
        else:
            norm_dx, norm_dy = ghost_length, ghost_length
        
        shadow_x = self.last_x + norm_dx
        shadow_y = self.last_y + norm_dy
        
        self.canvas.create_image(shadow_x - pad, shadow_y - pad, image=self.tk_shadow, anchor="nw", tag="ghost")
        self.canvas.create_image(self.last_x, self.last_y, image=self.tk_preview, tag="ghost")
        
        poly = get_grid_polygon(self.current_grid_shape, scaled_panel.width, scaled_panel.height)
        if poly:
            max_radius = max(math.hypot(x, y) for x, y in poly)
            self.canvas.create_oval(
                self.last_x - max_radius, self.last_y - max_radius,
                self.last_x + max_radius, self.last_y + max_radius,
                outline="#00ffff", dash=(4, 4), width=1, tags="ghost"
            )

    def on_move(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.draw_ghost()

    def on_space(self, event):
        if not self.work_img or not self.preview_panel_img: return
        self.roll_new_geometry()
        self.draw_ghost()

    def on_click(self, event, is_grid):
        if not self.work_img or not self.preview_panel_img: return
        
        cx, cy = event.x, event.y
        self.last_x, self.last_y = cx, cy
        
        scale_factor = self.current_size / max(self.preview_panel_img.size)
        cell_w = max(2, int(self.preview_panel_img.width * scale_factor))
        cell_w = (cell_w // 2) * 2
        cell_h = max(2, int(self.preview_panel_img.height * scale_factor))
        cell_h = (cell_h // 2) * 2
        
        if is_grid:
            grid_shape_to_save = self.current_grid_shape
        else:
            grid_shape_to_save = [(0, 0)]
            
        poly = get_grid_polygon(grid_shape_to_save, cell_w, cell_h)
        max_radius = max(math.hypot(x, y) for x, y in poly) if poly else 0
            
        self.placements.append({
            'cx': cx, 
            'cy': cy, 
            'grid_shape': grid_shape_to_save,
            'cell_w': cell_w,
            'cell_h': cell_h,
            'size': self.current_size,
            'max_radius': max_radius
        })

        scaled_panel = self._get_scaled_preview_panel()
        comp, _, _ = create_grid_composite(scaled_panel, grid_shape_to_save)
        
        img_geom = apply_geometry(comp, 0, 1.0, 1.0)
        pw, ph = img_geom.size
        px, py = int(cx - pw / 2), int(cy - ph / 2)
        
        auto_bright = calculate_brightness_adjustment(self.base_img, img_geom, px, py)
        img_rot = apply_photometry(img_geom, auto_bright, 0, 0.0)
        
        sh_opacity = calculate_local_shadow_opacity(self.base_img, px, py, pw, ph)
        
        shadow_img, pad = create_shadow(img_rot, blur_radius=self.ui_shadow_blur, opacity=sh_opacity)
        
        shadow_px = px + self.ui_shadow_dx - pad
        shadow_py = py + self.ui_shadow_dy - pad
        
        self.work_img.paste(shadow_img, (int(shadow_px), int(shadow_py)), mask=shadow_img)
        self.work_img.paste(img_rot, (px, py), mask=img_rot)
        
        self.redraw()
        self.roll_new_geometry()
        self.draw_ghost() 
        self.lbl_status.config(text=f"Dodano pozycję. Razem obiektów (masek): {len(self.placements)}")

    def redraw(self):
        if not self.work_img: return
        self.tk_bg = ImageTk.PhotoImage(self.work_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_bg, anchor="nw")
        
        for p in self.placements:
            cx, cy = p['cx'], p['cy']
            r = p.get('max_radius', 0)
            if r > 0:
                self.canvas.create_oval(
                    cx - r, cy - r,
                    cx + r, cy + r,
                    outline="#00ffff", dash=(4, 4), width=1, tags="overlay"
                )

    def save_batch(self):
        if not self.placements or not self.panel_images: 
            messagebox.showwarning("Brak danych", "Dodaj co najmniej 1 obiekt przed zapisem.")
            return
        
        self.lbl_status.config(text="Generowanie batcha... Proszę czekać.")
        self.root.update()
        
        try:
            current_bg_path = self.bg_images[self.current_bg_idx]
            original_filename = os.path.splitext(os.path.basename(current_bg_path))[0]

            # 1. Negative Sample
            if current_bg_path not in self.saved_empty_bgs:
                empty_base_name = f"{original_filename}_empty"
                self.base_img.convert("RGB").save(os.path.join(IMG_DIR, f"{empty_base_name}.jpg"))
                open(os.path.join(LBL_DIR, f"{empty_base_name}.txt"), "w").close() 
                
                empty_aug_data = {"global_augmentations": {}, "panels": []}
                with open(os.path.join(META_DIR, f"{empty_base_name}.json"), "w") as f:
                    json.dump(empty_aug_data, f, indent=4)
                    
                self.saved_empty_bgs.add(current_bg_path)

            # 2. Base Panels Variants
            for p_idx, panel_path in enumerate(self.panel_images):
                panel_img = Image.open(panel_path).convert("RGBA")
                out_img = self.base_img.copy() 
                labels_poly = []
                panel_augs = []
                
                # Roll shadow properties that stay consistent for this ENTIRE output variant image
                var_angle = random.uniform(0, 2 * math.pi)
                var_sh_len = random.uniform(SHADOW_LENGTH_MIN, SHADOW_LENGTH_MAX)
                var_sh_dx = math.cos(var_angle) * var_sh_len
                var_sh_dy = math.sin(var_angle) * var_sh_len
                var_sh_blur = random.uniform(SHADOW_BLUR_MIN, SHADOW_BLUR_MAX)
                
                for p in self.placements:
                    grid_shape = p['grid_shape']
                    cell_w = p['cell_w']
                    cell_h = p['cell_h']
                    
                    auto_rot = random.randint(ROT_MIN, ROT_MAX)
                    auto_noise = random.randint(NOISE_MIN, NOISE_MAX)
                    auto_stretch_x = random.uniform(STRETCH_MIN, STRETCH_MAX)
                    auto_stretch_y = random.uniform(STRETCH_MIN, STRETCH_MAX)
                    auto_blur = random.uniform(BLUR_MIN, BLUR_MAX)
                    
                    scaled_panel = panel_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                    
                    comp, cw, ch = create_grid_composite(scaled_panel, grid_shape)
                    
                    img_geom = apply_geometry(comp, auto_rot, auto_stretch_x, auto_stretch_y)
                    
                    pw_comp, ph_comp = img_geom.size
                    px, py = int(p['cx'] - pw_comp / 2), int(p['cy'] - ph_comp / 2)
                    auto_bright = calculate_brightness_adjustment(self.base_img, img_geom, px, py)
                    
                    img_rot = apply_photometry(img_geom, auto_bright, auto_noise, auto_blur)
                    
                    sh_opacity = calculate_local_shadow_opacity(self.base_img, px, py, pw_comp, ph_comp)
                    
                    shadow_img, pad = create_shadow(img_rot, blur_radius=var_sh_blur, opacity=sh_opacity)
                    
                    shadow_px = px + var_sh_dx - pad
                    shadow_py = py + var_sh_dy - pad
                    
                    out_img.paste(shadow_img, (int(shadow_px), int(shadow_py)), mask=shadow_img)
                    out_img.paste(img_rot, (px, py), mask=img_rot)
                    
                    base_poly = get_grid_polygon(grid_shape, cell_w, cell_h)
                    trans_poly = transform_polygon(base_poly, p['cx'], p['cy'], auto_rot, auto_stretch_x, auto_stretch_y)
                    norm_poly = normalize_polygon(trans_poly, out_img.width, out_img.height)
                    labels_poly.append((0, norm_poly))
                    
                    panel_augs.append({
                        "class_id": 0,
                        "grid_cells": len(grid_shape),
                        "locked_size": p['size'],
                        "rotation_deg": auto_rot,
                        "stretch_x": round(float(auto_stretch_x), 3),
                        "stretch_y": round(float(auto_stretch_y), 3),
                        "noise_intensity": auto_noise,
                        "blur_radius": round(float(auto_blur), 3),
                        "brightness_match_multiplier": round(float(auto_bright), 3),
                        "shadow": {
                            "offset_x": round(float(var_sh_dx), 3),
                            "offset_y": round(float(var_sh_dy), 3),
                            "blur_radius": round(float(var_sh_blur), 3),
                            "opacity": round(float(sh_opacity), 3)
                        }
                    })
                
                base_name = f"{original_filename}_pnl{p_idx}"
                out_img.convert("RGB").save(os.path.join(IMG_DIR, f"{base_name}.jpg"), quality=95)
                with open(os.path.join(LBL_DIR, f"{base_name}.txt"), "w") as f:
                    for cls, pts in labels_poly:
                        coords = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts])
                        f.write(f"{cls} {coords}\n")
                        
                base_aug_data = {"global_augmentations": {}, "panels": panel_augs}
                with open(os.path.join(META_DIR, f"{base_name}.json"), "w") as f:
                    json.dump(base_aug_data, f, indent=4)
                
                generate_dataset_variants(out_img, labels_poly, panel_augs, base_name)
                
            # 3. MIX Variant
            out_img_mix = self.base_img.copy()
            labels_poly_mix = []
            panel_augs_mix = []
            
            # Roll fresh consistent shadow properties for the MIX variant image
            mix_angle = random.uniform(0, 2 * math.pi)
            mix_sh_len = random.uniform(SHADOW_LENGTH_MIN, SHADOW_LENGTH_MAX)
            mix_sh_dx = math.cos(mix_angle) * mix_sh_len
            mix_sh_dy = math.sin(mix_angle) * mix_sh_len
            mix_sh_blur = random.uniform(SHADOW_BLUR_MIN, SHADOW_BLUR_MAX)
            
            for p in self.placements:
                random_panel_path = random.choice(self.panel_images)
                panel_img_mix = Image.open(random_panel_path).convert("RGBA")
                
                grid_shape = p['grid_shape']
                cell_w = p['cell_w']
                cell_h = p['cell_h']
                
                auto_rot = random.randint(ROT_MIN, ROT_MAX)
                auto_noise = random.randint(NOISE_MIN, NOISE_MAX)
                auto_stretch_x = random.uniform(STRETCH_MIN, STRETCH_MAX)
                auto_stretch_y = random.uniform(STRETCH_MIN, STRETCH_MAX)
                auto_blur = random.uniform(BLUR_MIN, BLUR_MAX)
                
                scaled_panel_mix = panel_img_mix.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                
                comp_mix, cw_mix, ch_mix = create_grid_composite(scaled_panel_mix, grid_shape)
                
                img_geom_mix = apply_geometry(comp_mix, auto_rot, auto_stretch_x, auto_stretch_y)
                
                pw_comp_mix, ph_comp_mix = img_geom_mix.size
                px_mix, py_mix = int(p['cx'] - pw_comp_mix / 2), int(p['cy'] - ph_comp_mix / 2)
                
                auto_bright_mix = calculate_brightness_adjustment(self.base_img, img_geom_mix, px_mix, py_mix)
                img_rot_mix = apply_photometry(img_geom_mix, auto_bright_mix, auto_noise, auto_blur)
                
                sh_opacity_mix = calculate_local_shadow_opacity(self.base_img, px_mix, py_mix, pw_comp_mix, ph_comp_mix)
                
                shadow_img_mix, pad_mix = create_shadow(img_rot_mix, blur_radius=mix_sh_blur, opacity=sh_opacity_mix)
                
                shadow_px_mix = px_mix + mix_sh_dx - pad_mix
                shadow_py_mix = py_mix + mix_sh_dy - pad_mix
                
                out_img_mix.paste(shadow_img_mix, (int(shadow_px_mix), int(shadow_py_mix)), mask=shadow_img_mix)
                out_img_mix.paste(img_rot_mix, (px_mix, py_mix), mask=img_rot_mix)
                
                base_poly_mix = get_grid_polygon(grid_shape, cell_w, cell_h)
                trans_poly_mix = transform_polygon(base_poly_mix, p['cx'], p['cy'], auto_rot, auto_stretch_x, auto_stretch_y)
                norm_poly_mix = normalize_polygon(trans_poly_mix, out_img_mix.width, out_img_mix.height)
                labels_poly_mix.append((0, norm_poly_mix))
                
                panel_augs_mix.append({
                    "class_id": 0,
                    "grid_cells": len(grid_shape),
                    "locked_size": p['size'],
                    "rotation_deg": auto_rot,
                    "stretch_x": round(float(auto_stretch_x), 3),
                    "stretch_y": round(float(auto_stretch_y), 3),
                    "noise_intensity": auto_noise,
                    "blur_radius": round(float(auto_blur), 3),
                    "brightness_match_multiplier": round(float(auto_bright_mix), 3),
                    "shadow": {
                        "offset_x": round(float(mix_sh_dx), 3),
                        "offset_y": round(float(mix_sh_dy), 3),
                        "blur_radius": round(float(mix_sh_blur), 3),
                        "opacity": round(float(sh_opacity_mix), 3)
                    }
                })
                
            base_name_mix = f"{original_filename}_mix"
            out_img_mix.convert("RGB").save(os.path.join(IMG_DIR, f"{base_name_mix}.jpg"))
            with open(os.path.join(LBL_DIR, f"{base_name_mix}.txt"), "w") as f:
                for cls, pts in labels_poly_mix:
                    coords = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts])
                    f.write(f"{cls} {coords}\n")
                    
            mix_aug_data = {"global_augmentations": {}, "panels": panel_augs_mix}
            with open(os.path.join(META_DIR, f"{base_name_mix}.json"), "w") as f:
                json.dump(mix_aug_data, f, indent=4)
                    
            generate_dataset_variants(out_img_mix, labels_poly_mix, panel_augs_mix, base_name_mix)
                
            self.lbl_status.config(text="Zapisano pomyślnie. Ładowanie kolejnego tła...")
            
            self.bg_images.pop(self.current_bg_idx)
            
            if self.bg_images:
                self.current_bg_idx = self.current_bg_idx % len(self.bg_images)
                self.root.after(11, self.load_current_bg) 
            else:
                self.canvas.delete("all")
                self.lbl_bg_info.config(text="Tło: 0/0")
                messagebox.showinfo("Koniec", "Wszystkie tła zostały przetworzone!")
            
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloObbApp(root)
    root.mainloop()