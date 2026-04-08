import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
import os
import uuid
import math
import random

OUTPUT_BASE_DIR = "dataset_yolo_seg26"
IMG_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

def overlaps(c1, c2):
    """Checks if two 2x2 logical cells overlap."""
    return abs(c1[0] - c2[0]) < 2 and abs(c1[1] - c2[1]) < 2

def generate_random_grid(num_cells):
    """Generates a random contiguous staggered shape (polyomino with half-shifts)."""
    if num_cells <= 1: return [(0, 0)]
    cells = [(0, 0)]
    
    # Valid adjacency moves for a 2x2 logical cell allowing half-shifts
    valid_moves = [
        (0, -2), (-1, -2), (1, -2), # Up (flush, left half, right half)
        (0, 2), (-1, 2), (1, 2),    # Down (flush, left half, right half)
        (-2, 0), (-2, -1), (-2, 1), # Left (flush, up half, down half)
        (2, 0), (2, -1), (2, 1)     # Right (flush, up half, down half)
    ]
    
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
    """Creates a transparent image containing the tightly packed, staggered grid."""
    # panel_img is guaranteed to be even width and height at this stage
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
    """Traces the outer boundary of the staggered grid to create a continuous mask."""
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
        
        # Break all 4 edges of the cell into 8 half-edges for perfect cancellation
        cell_edges = [
            ((cx, cy), (cx+1, cy)),       # Top 1
            ((cx+1, cy), (cx+2, cy)),     # Top 2
            ((cx+2, cy), (cx+2, cy+1)),   # Right 1
            ((cx+2, cy+1), (cx+2, cy+2)), # Right 2
            ((cx+2, cy+2), (cx+1, cy+2)), # Bottom 1
            ((cx+1, cy+2), (cx, cy+2)),   # Bottom 2
            ((cx, cy+2), (cx, cy+1)),     # Left 1
            ((cx, cy+1), (cx, cy))        # Left 2
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

def apply_noise_np(img, intensity=20):
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

def create_shadow(img, blur_radius, opacity):
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
    mask = alpha > 10 
    
    if not np.any(mask): return 1.0
        
    def get_lum(arr):
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        
    bg_lum = np.mean(get_lum(bg_arr)[mask])
    panel_lum = np.mean(get_lum(panel_arr)[mask])
    
    if panel_lum < 1.0: panel_lum = 1.0 
    
    factor = bg_lum / panel_lum
    return max(0.2, min(factor, 3.0))

def rotate_point(x, y, cx, cy, angle_rad):
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + cx, ry + cy

def generate_dataset_variants(original_img, original_labels_poly, base_name):
    w_img, h_img = original_img.size
    
    img_aug = original_img.copy()
    labels_aug = [(cls, list(pts)) for cls, pts in original_labels_poly]

    img_aug = apply_noise_np(img_aug, random.randint(5, 11))

    angle = random.choice([random.randint(-15, -5), random.randint(5, 15)])
    
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

    f_name = f"{base_name}_aug1"
    img_aug.convert("RGB").save(os.path.join(IMG_DIR, f"{f_name}.jpg"), quality=95)
    
    with open(os.path.join(LBL_DIR, f"{f_name}.txt"), "w") as f:
        for cls, pts in labels_aug:
            coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
            f.write(f"{cls} {coords}\n")

# --- GUI APPLICATION ---

class YoloObbApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Seg Batch Generator (Staggered Half-Shift Grids)")
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
        
        self.current_grid_shape = generate_random_grid(random.randint(2, 6))
        self.current_size = random.randint(22, 44)

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
        
        tk.Button(ctrl, text="Pomiń Tło (Bez zapisu) ->", command=self.skip_bg, bg="#ff9999").pack(fill=tk.X, pady=5)

        tk.Label(ctrl, text="--- Parametry Generacji ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        
        info_font = ("Arial", 9, "bold")
        info_color = "#2e7d32"
        tk.Label(ctrl, text="Rozmiar Pixeli i Kształt: ZABLOKOWANE DLA KLIKU", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="Anizotropia (Skrót osi): LOSOWA PER PANEL", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="Rozmycie (Blur): LOSOWE PER PANEL", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="Obrót: LOSOWY PER PANEL", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        
        tk.Label(ctrl, text="Cień (Shadow): ZMIENNY KIERUNEK", bg="#dddddd", fg="#b71c1c", font=("Arial", 9, "bold", "italic")).pack(anchor="w", pady=(8,2))
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
        folder = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26'
        if folder:
            self.bg_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.bg_images:
                self.current_bg_idx = 0
                self.load_current_bg()
            else:
                messagebox.showwarning("Pusto", "Brak obrazów w folderze.")

    def load_panel_folder(self):
        folder = './pvs'
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
        self.reset_canvas()
        self.check_ready_state()

    def next_bg(self):
        if not self.bg_images: return
        self.current_bg_idx = (self.current_bg_idx + 1) % len(self.bg_images)
        self.load_current_bg()
        
    def skip_bg(self):
        self.lbl_status.config(text="Pominięto tło.")
        self.next_bg()

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
        self.current_grid_shape = generate_random_grid(random.randint(2, 6))
        self.current_size = random.randint(22, 44)

    def _get_scaled_preview_panel(self):
        scale_factor = self.current_size / max(self.preview_panel_img.size)
        scaled_w = max(2, int(self.preview_panel_img.width * scale_factor))
        scaled_w = (scaled_w // 2) * 2 # Force even
        scaled_h = max(2, int(self.preview_panel_img.height * scale_factor))
        scaled_h = (scaled_h // 2) * 2 # Force even
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
        
        shadow_img, pad = create_shadow(processed_ov, blur_radius=4.0, opacity=0.35)
        
        self.tk_shadow = ImageTk.PhotoImage(shadow_img)
        self.tk_preview = ImageTk.PhotoImage(processed_ov)
        
        shadow_offset_x = 4
        shadow_offset_y = 4
        
        self.canvas.create_image(self.last_x + shadow_offset_x, self.last_y + shadow_offset_y, image=self.tk_shadow, tag="ghost")
        self.canvas.create_image(self.last_x, self.last_y, image=self.tk_preview, tag="ghost")

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
            
        self.placements.append({
            'cx': cx, 
            'cy': cy, 
            'grid_shape': grid_shape_to_save,
            'cell_w': cell_w,
            'cell_h': cell_h
        })

        scaled_panel = self._get_scaled_preview_panel()
        comp, _, _ = create_grid_composite(scaled_panel, grid_shape_to_save)
        
        img_geom = apply_geometry(comp, 0, 1.0, 1.0)
        pw, ph = img_geom.size
        px, py = int(cx - pw / 2), int(cy - ph / 2)
        
        auto_bright = calculate_brightness_adjustment(self.base_img, img_geom, px, py)
        img_rot = apply_photometry(img_geom, auto_bright, 0, 0.0)
        
        shadow_img, pad = create_shadow(img_rot, blur_radius=4.0, opacity=0.35)
        shadow_px = px + 4 - pad
        shadow_py = py + 4 - pad
        self.work_img.paste(shadow_img, (shadow_px, shadow_py), mask=shadow_img)
        
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

    def save_batch(self):
        if not self.placements or not self.panel_images: 
            messagebox.showwarning("Brak danych", "Dodaj co najmniej 1 obiekt przed zapisem.")
            return
        
        self.lbl_status.config(text="Generowanie batcha... Proszę czekać.")
        self.root.update()
        
        try:
            # 1. Negative Sample
            empty_base_name = f"bg{self.current_bg_idx}_empty_{uuid.uuid4().hex[:4]}"
            self.base_img.convert("RGB").save(os.path.join(IMG_DIR, f"{empty_base_name}.jpg"))
            open(os.path.join(LBL_DIR, f"{empty_base_name}.txt"), "w").close() 

            # 2. Base Panels Variants
            for p_idx, panel_path in enumerate(self.panel_images):
                panel_img = Image.open(panel_path).convert("RGBA")
                out_img = self.base_img.copy() 
                labels_poly = []
                
                for p in self.placements:
                    grid_shape = p['grid_shape']
                    cell_w = p['cell_w']
                    cell_h = p['cell_h']
                    
                    auto_rot = random.randint(-180, 180)
                    auto_noise = random.randint(0, 11)
                    auto_stretch_x = random.uniform(0.8, 1.2)
                    auto_stretch_y = random.uniform(0.8, 1.2)
                    auto_blur = random.uniform(0.0, 1.0)
                    
                    sh_off_x = random.randint(2, 6)
                    sh_off_y = random.randint(2, 6)
                    sh_blur = random.uniform(3.0, 6.0)
                    sh_opacity = random.uniform(0.2, 0.5)
                    
                    scaled_panel = panel_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                    
                    comp, cw, ch = create_grid_composite(scaled_panel, grid_shape)
                    
                    img_geom = apply_geometry(comp, auto_rot, auto_stretch_x, auto_stretch_y)
                    
                    pw_comp, ph_comp = img_geom.size
                    px, py = int(p['cx'] - pw_comp / 2), int(p['cy'] - ph_comp / 2)
                    auto_bright = calculate_brightness_adjustment(self.base_img, img_geom, px, py)
                    
                    img_rot = apply_photometry(img_geom, auto_bright, auto_noise, auto_blur)
                    
                    shadow_img, pad = create_shadow(img_rot, sh_blur, sh_opacity)
                    out_img.paste(shadow_img, (px + sh_off_x - pad, py + sh_off_y - pad), mask=shadow_img)
                    out_img.paste(img_rot, (px, py), mask=img_rot)
                    
                    base_poly = get_grid_polygon(grid_shape, cell_w, cell_h)
                    trans_poly = transform_polygon(base_poly, p['cx'], p['cy'], auto_rot, auto_stretch_x, auto_stretch_y)
                    norm_poly = normalize_polygon(trans_poly, out_img.width, out_img.height)
                    labels_poly.append((0, norm_poly))
                
                base_name = f"bg{self.current_bg_idx}_pnl{p_idx}_{uuid.uuid4().hex[:4]}"
                out_img.convert("RGB").save(os.path.join(IMG_DIR, f"{base_name}.jpg"))
                with open(os.path.join(LBL_DIR, f"{base_name}.txt"), "w") as f:
                    for cls, pts in labels_poly:
                        coords = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts])
                        f.write(f"{cls} {coords}\n")
                
                generate_dataset_variants(out_img, labels_poly, base_name)
                
            # 3. MIX Variant
            out_img_mix = self.base_img.copy()
            labels_poly_mix = []
            
            for p in self.placements:
                random_panel_path = random.choice(self.panel_images)
                panel_img_mix = Image.open(random_panel_path).convert("RGBA")
                
                grid_shape = p['grid_shape']
                cell_w = p['cell_w']
                cell_h = p['cell_h']
                
                auto_rot = random.randint(-180, 180)
                auto_noise = random.randint(0, 11)
                auto_stretch_x = random.uniform(0.8, 1.2)
                auto_stretch_y = random.uniform(0.8, 1.2)
                auto_blur = random.uniform(0.0, 1.0)
                
                sh_off_x = random.randint(2, 6)
                sh_off_y = random.randint(2, 6)
                sh_blur = random.uniform(3.0, 6.0)
                sh_opacity = random.uniform(0.2, 0.5)
                
                scaled_panel_mix = panel_img_mix.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                
                comp_mix, cw_mix, ch_mix = create_grid_composite(scaled_panel_mix, grid_shape)
                
                img_geom_mix = apply_geometry(comp_mix, auto_rot, auto_stretch_x, auto_stretch_y)
                
                pw_comp_mix, ph_comp_mix = img_geom_mix.size
                px_mix, py_mix = int(p['cx'] - pw_comp_mix / 2), int(p['cy'] - ph_comp_mix / 2)
                
                auto_bright_mix = calculate_brightness_adjustment(self.base_img, img_geom_mix, px_mix, py_mix)
                img_rot_mix = apply_photometry(img_geom_mix, auto_bright_mix, auto_noise, auto_blur)
                
                shadow_img_mix, pad_mix = create_shadow(img_rot_mix, sh_blur, sh_opacity)
                out_img_mix.paste(shadow_img_mix, (px_mix + sh_off_x - pad_mix, py_mix + sh_off_y - pad_mix), mask=shadow_img_mix)
                out_img_mix.paste(img_rot_mix, (px_mix, py_mix), mask=img_rot_mix)
                
                base_poly_mix = get_grid_polygon(grid_shape, cell_w, cell_h)
                trans_poly_mix = transform_polygon(base_poly_mix, p['cx'], p['cy'], auto_rot, auto_stretch_x, auto_stretch_y)
                norm_poly_mix = normalize_polygon(trans_poly_mix, out_img_mix.width, out_img_mix.height)
                labels_poly_mix.append((0, norm_poly_mix))
                
            base_name_mix = f"bg{self.current_bg_idx}_mix_{uuid.uuid4().hex[:4]}"
            out_img_mix.convert("RGB").save(os.path.join(IMG_DIR, f"{base_name_mix}.jpg"))
            with open(os.path.join(LBL_DIR, f"{base_name_mix}.txt"), "w") as f:
                for cls, pts in labels_poly_mix:
                    coords = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts])
                    f.write(f"{cls} {coords}\n")
                    
            generate_dataset_variants(out_img_mix, labels_poly_mix, base_name_mix)
                
            self.lbl_status.config(text="Zapisano pomyślnie. Ładowanie kolejnego tła...")
            self.root.after(77, self.next_bg) 
            
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloObbApp(root)
    root.mainloop()