import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import os
import uuid
import math
import random

OUTPUT_BASE_DIR = "dataset_yolo_seg"
IMG_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

def generate_random_grid(num_cells):
    """Generates a random contiguous Tetris-like shape (polyomino)."""
    if num_cells <= 1: return [(0, 0)]
    cells = {(0, 0)}
    adj = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    
    for _ in range(num_cells - 1):
        if not adj: break
        new_cell = random.choice(list(adj))
        cells.add(new_cell)
        adj.remove(new_cell)
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (new_cell[0] + dx, new_cell[1] + dy)
            if neighbor not in cells:
                adj.add(neighbor)
                
    return list(cells)

def create_grid_composite(panel_img, grid_shape):
    """Creates a transparent image containing the unrotated grid."""
    min_x = min(gx for gx, gy in grid_shape)
    max_x = max(gx for gx, gy in grid_shape)
    min_y = min(gy for gx, gy in grid_shape)
    max_y = max(gy for gx, gy in grid_shape)
    
    w, h = panel_img.size
    gw = (max_x - min_x + 1) * w
    gh = (max_y - min_y + 1) * h
    
    composite = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
    for gx, gy in grid_shape:
        px = (gx - min_x) * w
        py = (gy - min_y) * h
        composite.paste(panel_img, (px, py))
        
    return composite

def get_grid_polygon(grid_shape, cell_w, cell_h):
    """
    Traces the outer boundary of the grid to create a continuous polygon mask.
    Returns coordinates relative to the center of the grid composite.
    """
    min_x = min(gx for gx, gy in grid_shape)
    min_y = min(gy for gx, gy in grid_shape)
    max_x = max(gx for gx, gy in grid_shape)
    max_y = max(gy for gx, gy in grid_shape)
    
    edges = set()
    for gx, gy in grid_shape:
        cx = gx - min_x
        cy = gy - min_y
        
        # Directed edges for a cell
        cell_edges = [
            ((cx, cy), (cx+1, cy)),       # Top
            ((cx+1, cy), (cx+1, cy+1)),   # Right
            ((cx+1, cy+1), (cx, cy+1)),   # Bottom
            ((cx, cy+1), (cx, cy))        # Left
        ]
        
        for edge in cell_edges:
            reverse_edge = (edge[1], edge[0])
            if reverse_edge in edges:
                edges.remove(reverse_edge) # Internal edge shared by two cells
            else:
                edges.add(edge)
                
    # Link edges to form the continuous boundary
    adj = {start: end for start, end in edges}
    
    start_node = next(iter(adj.keys()))
    polygon = []
    current_node = start_node
    
    while True:
        polygon.append(current_node)
        current_node = adj[current_node]
        if current_node == start_node:
            break
            
    # Center the polygon relative to the composite image size
    comp_w_cells = (max_x - min_x + 1)
    comp_h_cells = (max_y - min_y + 1)
    center_x = comp_w_cells / 2.0
    center_y = comp_h_cells / 2.0
    
    centered_poly = []
    for nx, ny in polygon:
        px = (nx - center_x) * cell_w
        py = (ny - center_y) * cell_h
        centered_poly.append((px, py))
        
    return centered_poly

def transform_polygon(polygon, cx, cy, rot_deg, shear_x, shear_y):
    """Applies shear, rotation, and translation to a continuous polygon mask."""
    angle_rad = math.radians(-rot_deg)
    transformed = []
    for x, y in polygon:
        # Shear
        sx = x + shear_x * y
        sy = y + shear_y * x
        
        # Rotate
        rx = sx * math.cos(angle_rad) - sy * math.sin(angle_rad)
        ry = sx * math.sin(angle_rad) + sy * math.cos(angle_rad)
        
        transformed.append((cx + rx, cy + ry))
    return transformed

def normalize_polygon(polygon, img_w, img_h):
    """Normalizes polygon coordinates between 0 and 1."""
    norm_poly = []
    for x, y in polygon:
        nx = max(0, min(img_w, x)) / img_w 
        ny = max(0, min(img_h, y)) / img_h 
        norm_poly.append((nx, ny))
    return norm_poly

def rotate_point(x, y, cx, cy, angle_rad):
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + cx, ry + cy

def apply_noise_np(img, intensity=20):
    if intensity <= 0: return img
    arr = np.array(img)
    noise = np.random.normal(0, intensity, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy, mode=img.mode)

def apply_shear_to_image(img, shear_x, shear_y):
    """Applies affine shear to a PIL image, expanding the canvas so it isn't cropped."""
    w, h = img.size
    
    # Corners of the source image relative to its center
    corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    
    # Calculate where the corners will end up after shear
    sheared_corners = [(x + shear_x * y, y + shear_y * x) for x, y in corners]
    
    # Calculate new bounding box needed to fit the sheared image
    min_x = min(c[0] for c in sheared_corners)
    max_x = max(c[0] for c in sheared_corners)
    min_y = min(c[1] for c in sheared_corners)
    max_y = max(c[1] for c in sheared_corners)
    
    new_w = int(math.ceil(max_x - min_x))
    new_h = int(math.ceil(max_y - min_y))
    
    # Calculate inverse affine transform matrix for PIL
    det = 1 - shear_x * shear_y
    if abs(det) < 0.0001: det = 0.0001 # Prevent division by zero just in case
    
    inv_a = 1 / det
    inv_b = -shear_x / det
    inv_c = -shear_y / det
    inv_d = 1 / det
    
    # Calculate translations to keep the image perfectly centered
    tx = -inv_a * (new_w/2) - inv_b * (new_h/2) + w/2
    ty = -inv_c * (new_w/2) - inv_d * (new_h/2) + h/2
    
    return img.transform(
        (new_w, new_h), 
        Image.AFFINE, 
        (inv_a, inv_b, tx, inv_c, inv_d, ty), 
        resample=Image.BICUBIC
    )

def generate_dataset_variants(original_img, original_labels_poly, base_name):
    w_img, h_img = original_img.size
    
    # Changed to 2 variants since crop was removed
    for i in range(1, 3):
        img_aug = original_img.copy()
        labels_aug = [(cls, list(pts)) for cls, pts in original_labels_poly]

        # 1. Rotation 
        angle = random.randint(-15, 15)
        if angle != 0:
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

        # 2. Noise (70% chance)
        if random.random() > 0.3:
            img_aug = apply_noise_np(img_aug, random.randint(10, 40))

        # Save variant
        f_name = f"{base_name}_aug{i}"
        img_aug.convert("RGB").save(os.path.join(IMG_DIR, f"{f_name}.jpg"), quality=95)
        
        with open(os.path.join(LBL_DIR, f"{f_name}.txt"), "w") as f:
            for cls, pts in labels_aug:
                coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
                f.write(f"{cls} {coords}\n")

# --- GUI APPLICATION ---

class YoloObbApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Segmentation Generator (Auto + Shear + Mix + Masks)")
        self.root.geometry("1280x800")

        # Folder management
        self.bg_images = []
        self.panel_images = []
        self.current_bg_idx = 0
        
        self.base_img = None
        self.work_img = None
        self.preview_panel_img = None
        
        # Stores dictionaries of placements: {cx, cy, scale, rot, bright, noise}
        self.placements = [] 
        self.tk_preview = None
        
        # Track coordinates for the live ghost drawing
        self.last_x = 0
        self.last_y = 0
        # Pre-generate the first grid shape to display as the cursor preview
        self.current_grid_shape = generate_random_grid(random.randint(2, 6))

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
        
        # Added skip button
        tk.Button(ctrl, text="Pomiń Tło (Bez zapisu) ->", command=self.skip_bg, bg="#ff9999").pack(fill=tk.X, pady=5)

        tk.Label(ctrl, text="--- Parametry Generacji ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        
        # UI updated to reflect full automation
        info_font = ("Arial", 9, "bold")
        info_color = "#2e7d32"
        tk.Label(ctrl, text="Rozmiar 1 Panelu: LOSOWY (30-100px)", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="Ścinanie (Shear): LOSOWE (-0.3 do 0.3)", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="Obrót: LOSOWY (-180° do 180°)", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="Jasność: LOSOWA (0.5x - 1.5x)", bg="#dddddd", fg=info_color, font=info_font).pack(anchor="w", pady=2)
        
        tk.Label(ctrl, text="--- Sterowanie ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        tk.Label(ctrl, text="LEWY KLIK: Postaw Grid z podglądu", bg="#dddddd", fg="#b71c1c", font=("Arial", 9, "bold")).pack(anchor="w", pady=2)
        tk.Label(ctrl, text="PRAWY KLIK: Postaw Pojedynczy Panel", bg="#dddddd", fg="#0d47a1", font=("Arial", 9, "bold")).pack(anchor="w", pady=2)

        tk.Label(ctrl, text="--- Zapis ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        self.btn_save = tk.Button(ctrl, text="ZAPISZ WSZYSTKIE WARIANTY", command=self.save_batch, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=10, ipady=5)
        
        tk.Button(ctrl, text="Wyczyść kliknięcia", command=self.reset_canvas).pack(fill=tk.X)
        
        self.lbl_status = tk.Label(ctrl, text="Gotowy", bg="#dddddd", fg="blue")
        self.lbl_status.pack(side=tk.BOTTOM, pady=10)

        # Canvas
        self.cv_frame = tk.Frame(self.root, bg="#333")
        self.cv_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(self.cv_frame, bg="#333", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", lambda e: self.on_click(e, is_grid=True))
        self.canvas.bind("<Button-2>", lambda e: self.on_click(e, is_grid=False)) 
        self.canvas.bind("<Button-3>", lambda e: self.on_click(e, is_grid=False))
        
        self.canvas.bind("<Motion>", self.on_move)

    def load_bg_folder(self):
        # folder = filedialog.askdirectory(title="Wybierz folder z tłami")
        folder = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26'
        if folder:
            self.bg_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.bg_images:
                self.current_bg_idx = 0
                self.load_current_bg()
            else:
                messagebox.showwarning("Pusto", "Brak obrazów w folderze.")

    def load_panel_folder(self):
        # folder = filedialog.askdirectory(title="Wybierz folder z panelami")
        folder = './pvs'
        if folder:
            self.panel_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.panel_images:
                # Load the first panel just for the visual preview cursor
                self.preview_panel_img = Image.open(self.panel_images[0]).convert("RGBA")
                self.current_grid_shape = generate_random_grid(random.randint(2, 6))
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
        """Skips current background without saving any data."""
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

    def apply_transform(self, img, scale_factor, rot, bright, noise, shear_x, shear_y):
        base_w = max(1, int(img.width * scale_factor))
        base_h = max(1, int(img.height * scale_factor))
        res = img.resize((base_w, base_h), Image.Resampling.LANCZOS)
        
        if bright != 1.0: res = ImageEnhance.Brightness(res).enhance(bright)
        if noise > 0: res = apply_noise_np(res, noise)
        
        # 1. Apply Shear mapping
        if shear_x != 0.0 or shear_y != 0.0:
            res = apply_shear_to_image(res, shear_x, shear_y)
            
        # 2. Apply Rotation mapping
        res_rotated = res.rotate(rot, expand=True, resample=Image.BICUBIC)
        return res_rotated, base_w, base_h

    def draw_ghost(self):
        """Draws the current queued shape as a ghost following the mouse."""
        if not self.work_img or not self.preview_panel_img: return
        
        self.canvas.delete("ghost")
        
        # Build composite of the currently queued shape
        comp = create_grid_composite(self.preview_panel_img, self.current_grid_shape)
        scale_factor = 65 / max(self.preview_panel_img.size) 
        
        processed_ov, _, _ = self.apply_transform(comp, scale_factor, 0, 1.0, 0, 0.0, 0.0)
        self.tk_preview = ImageTk.PhotoImage(processed_ov)
        
        self.canvas.create_image(self.last_x, self.last_y, image=self.tk_preview, tag="ghost")

    def on_move(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.draw_ghost()

    def on_click(self, event, is_grid):
        if not self.work_img or not self.preview_panel_img: return
        
        cx, cy = event.x, event.y
        self.last_x, self.last_y = cx, cy
        
        if is_grid:
            # Save the current shape we were previewing
            grid_shape_to_save = self.current_grid_shape
            # Roll a new random shape for the NEXT placement
            self.current_grid_shape = generate_random_grid(random.randint(2, 6))
        else:
            # Right click forces a single panel
            grid_shape_to_save = [(0, 0)]
            
        self.placements.append({'cx': cx, 'cy': cy, 'grid_shape': grid_shape_to_save})

        # Visually stamp the shape onto the working canvas so we can see it
        comp = create_grid_composite(self.preview_panel_img, grid_shape_to_save)
        scale_factor = 65 / max(self.preview_panel_img.size)
        
        img_rotated, _, _ = self.apply_transform(comp, scale_factor, 0, 1.0, 0, 0.0, 0.0)
        paste_w, paste_h = img_rotated.size
        paste_x, paste_y = int(cx - paste_w / 2), int(cy - paste_h / 2)
        
        self.work_img.paste(img_rotated, (paste_x, paste_y), mask=img_rotated)
        self.redraw()
        self.draw_ghost() # Keep ghost active after stamp
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
            # 1. Zapis pustego tła (Negative Sample)
            empty_base_name = f"bg{self.current_bg_idx}_empty_{uuid.uuid4().hex[:4]}"
            self.base_img.convert("RGB").save(os.path.join(IMG_DIR, f"{empty_base_name}.jpg"))
            
            # Tworzenie pustego pliku txt
            open(os.path.join(LBL_DIR, f"{empty_base_name}.txt"), "w").close() 

            # 2. Zapis wariantów z pojedynczymi panelami
            for p_idx, panel_path in enumerate(self.panel_images):
                panel_img = Image.open(panel_path).convert("RGBA")
                out_img = self.base_img.copy() 
                labels_poly = []
                
                # Apply all saved placements to THIS specific panel
                for p in self.placements:
                    # ---> FULL AUTOMATION GENERATOR <---
                    auto_size = random.randint(30, 77)
                    auto_rot = random.randint(-180, 180)
                    auto_bright = random.uniform(0.5, 1.5)
                    auto_noise = random.randint(0, 50)
                    auto_shear_x = random.uniform(-0.3, 0.3)
                    auto_shear_y = random.uniform(-0.3, 0.3)
                    
                    grid_shape = p['grid_shape']
                    comp = create_grid_composite(panel_img, grid_shape)
                    scale_factor = auto_size / max(panel_img.size)
                    
                    img_rot, comp_w, comp_h = self.apply_transform(
                        comp, scale_factor, auto_rot, auto_bright, auto_noise, auto_shear_x, auto_shear_y
                    )
                    
                    pw_comp, ph_comp = img_rot.size
                    px, py = int(p['cx'] - pw_comp / 2), int(p['cy'] - ph_comp / 2)
                    out_img.paste(img_rot, (px, py), mask=img_rot)
                    
                    # Compute continuous polygon mask for the joint object
                    pw_cell = panel_img.width * scale_factor
                    ph_cell = panel_img.height * scale_factor
                    
                    base_poly = get_grid_polygon(grid_shape, pw_cell, ph_cell)
                    trans_poly = transform_polygon(base_poly, p['cx'], p['cy'], auto_rot, auto_shear_x, auto_shear_y)
                    norm_poly = normalize_polygon(trans_poly, out_img.width, out_img.height)
                    labels_poly.append((0, norm_poly))
                
                base_name = f"bg{self.current_bg_idx}_pnl{p_idx}_{uuid.uuid4().hex[:4]}"
                
                # Save original combo
                out_img.convert("RGB").save(os.path.join(IMG_DIR, f"{base_name}.jpg"))
                with open(os.path.join(LBL_DIR, f"{base_name}.txt"), "w") as f:
                    for cls, pts in labels_poly:
                        coords = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts])
                        f.write(f"{cls} {coords}\n")
                
                generate_dataset_variants(out_img, labels_poly, base_name)
                
            # 3. Zapis wariantu "MIX" (różne panele na jednym zdjęciu)
            out_img_mix = self.base_img.copy()
            labels_poly_mix = []
            
            for p in self.placements:
                # Wybieramy losowy panel z dostępnych dla każdego kliknięcia z osobna
                random_panel_path = random.choice(self.panel_images)
                panel_img_mix = Image.open(random_panel_path).convert("RGBA")
                
                auto_size = random.randint(30, 100)
                auto_rot = random.randint(-180, 180)
                auto_bright = random.uniform(0.5, 1.5)
                auto_noise = random.randint(0, 50)
                auto_shear_x = random.uniform(-0.3, 0.3)
                auto_shear_y = random.uniform(-0.3, 0.3)
                
                grid_shape = p['grid_shape']
                comp_mix = create_grid_composite(panel_img_mix, grid_shape)
                scale_factor_mix = auto_size / max(panel_img_mix.size)
                
                img_rot_mix, _, _ = self.apply_transform(
                    comp_mix, scale_factor_mix, auto_rot, auto_bright, auto_noise, auto_shear_x, auto_shear_y
                )
                
                pw_comp_mix, ph_comp_mix = img_rot_mix.size
                px_mix, py_mix = int(p['cx'] - pw_comp_mix / 2), int(p['cy'] - ph_comp_mix / 2)
                out_img_mix.paste(img_rot_mix, (px_mix, py_mix), mask=img_rot_mix)
                
                # Compute continuous polygon mask for the MIX object
                pw_cell = panel_img_mix.width * scale_factor_mix
                ph_cell = panel_img_mix.height * scale_factor_mix
                
                base_poly_mix = get_grid_polygon(grid_shape, pw_cell, ph_cell)
                trans_poly_mix = transform_polygon(base_poly_mix, p['cx'], p['cy'], auto_rot, auto_shear_x, auto_shear_y)
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
            self.root.after(500, self.next_bg) 
            
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloObbApp(root)
    root.mainloop()