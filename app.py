import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import os
import uuid
import math
import random

OUTPUT_BASE_DIR = "dataset_yolo_obb"
IMG_DIR = os.path.join(OUTPUT_BASE_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_BASE_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

def calculate_obb_corners(cx, cy, w, h, angle_deg):
    angle_rad = math.radians(-angle_deg) 
    dw, dh = w / 2, h / 2
    
    corners_local = [(-dw, -dh), (dw, -dh), (dw, dh), (-dw, dh)]
    
    rotated_corners = []
    for x, y in corners_local:
        rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated_corners.append((cx + rx, cy + ry))
        
    return rotated_corners

def normalize_obb(corners, img_w, img_h):
    norm_corners = []
    for x, y in corners:
        nx = max(0, min(img_w, x)) / img_w 
        ny = max(0, min(img_h, y)) / img_h 
        norm_corners.append((nx, ny))
    return norm_corners

def rotate_point(x, y, cx, cy, angle_rad):
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + cx, ry + cy

def apply_noise_np(img, intensity=20):
    arr = np.array(img)
    noise = np.random.normal(0, intensity, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy, mode=img.mode)

def generate_dataset_variants(original_img, original_labels_obb, base_name):
    w_img, h_img = original_img.size
    
    for i in range(1, 4):
        img_aug = original_img.copy()
        labels_aug = [(cls, list(pts)) for cls, pts in original_labels_obb]

        # 1. Rotation
        if i in [1, 3]:
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

        # 2. Crop
        if i in [2, 3]:
            crop_p = 0.1
            cx = random.randint(0, int(w_img * crop_p))
            cy = random.randint(0, int(h_img * crop_p))
            cw = w_img - cx - random.randint(0, int(w_img * crop_p))
            ch = h_img - cy - random.randint(0, int(h_img * crop_p))
            
            img_aug = img_aug.crop((cx, cy, cx+cw, cy+ch))
            
            new_labels = []
            for cls, points in labels_aug:
                new_pts = []
                for nx, ny in points:
                    px, py = nx * w_img, ny * h_img
                    npx, npy = px - cx, py - cy
                    npx = max(0, min(cw, npx))
                    npy = max(0, min(ch, npy))
                    new_pts.append((npx / cw, npy / ch))
                new_labels.append((cls, new_pts))
            labels_aug = new_labels

        # 3. Noise
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
        self.root.title("YOLOv8 OBB Batch Generator")
        self.root.geometry("1280x800")

        self.bg_images = []
        self.panel_images = []
        self.current_bg_idx = 0
        
        self.base_img = None
        self.work_img = None
        self.preview_panel_img = None
        
        self.placements = [] 
        self.tk_preview = None

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

        tk.Label(ctrl, text="--- Parametry Panelu ---", bg="#dddddd", font=("Arial", 10, "bold")).pack(pady=(15,5))
        
        # Zmieniono ze "Skali (%)" na "Rozmiar w Pikselach" z limitem 30-100
        tk.Label(ctrl, text="Rozmiar Panelu (max px)", bg="#dddddd").pack(anchor="w")
        self.s_size = tk.Scale(ctrl, from_=30, to=100, orient="horizontal", bg="#dddddd")
        self.s_size.set(60)
        self.s_size.pack(fill=tk.X)

        tk.Label(ctrl, text="Obrót", bg="#dddddd", fg="red").pack(anchor="w")
        self.s_rot = tk.Scale(ctrl, from_=-180, to=180, orient="horizontal", bg="#dddddd")
        self.s_rot.set(0)
        self.s_rot.pack(fill=tk.X)

        tk.Label(ctrl, text="Jasność / Kontrast", bg="#dddddd").pack(anchor="w")
        self.s_bright = tk.Scale(ctrl, from_=0.5, to=1.5, resolution=0.1, orient="horizontal", label="Jasność", bg="#dddddd")
        self.s_bright.set(1.0)
        self.s_bright.pack(fill=tk.X)
        
        self.s_noise = tk.Scale(ctrl, from_=0, to=50, orient="horizontal", label="Szum", bg="#dddddd")
        self.s_noise.pack(fill=tk.X)
        
        tk.Button(ctrl, text="Reset Suwaków", command=self.reset_sliders).pack(fill=tk.X, pady=5)

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

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_move)

    def reset_sliders(self):
        self.s_size.set(60)
        self.s_rot.set(0)
        self.s_bright.set(1.0)
        self.s_noise.set(0)

    def load_bg_folder(self):
        folder = filedialog.askdirectory(title="Wybierz folder z tłami")
        if folder:
            self.bg_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.bg_images:
                self.current_bg_idx = 0
                self.load_current_bg()
            else:
                messagebox.showwarning("Pusto", "Brak obrazów w folderze.")

    def load_panel_folder(self):
        folder = filedialog.askdirectory(title="Wybierz folder z panelami")
        if folder:
            self.panel_images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if self.panel_images:
                self.preview_panel_img = Image.open(self.panel_images[0]).convert("RGBA")
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

    def apply_transform(self, img, target_size, rot, bright, noise):
        """Skaluje obraz tak, aby jego najdłuższy bok był równy target_size px"""
        # Obliczanie matematycznej proporcji skalowania
        max_dim = max(img.width, img.height)
        if max_dim == 0: max_dim = 1
        scale = target_size / max_dim
        
        base_w = max(1, int(img.width * scale))
        base_h = max(1, int(img.height * scale))
        res = img.resize((base_w, base_h), Image.Resampling.LANCZOS)
        
        if bright != 1.0: res = ImageEnhance.Brightness(res).enhance(bright)
        if noise > 0: res = apply_noise_np(res, noise)
        
        res_rotated = res.rotate(rot, expand=True, resample=Image.BICUBIC)
        return res_rotated, base_w, base_h

    def on_move(self, event):
        if not self.work_img or not self.preview_panel_img: return
        
        processed_ov, _, _ = self.apply_transform(
            self.preview_panel_img, 
            self.s_size.get(), self.s_rot.get(), 
            self.s_bright.get(), self.s_noise.get()
        )
        self.tk_preview = ImageTk.PhotoImage(processed_ov)
        
        self.canvas.delete("ghost")
        self.canvas.create_image(event.x, event.y, image=self.tk_preview, tag="ghost")

    def on_click(self, event):
        if not self.work_img or not self.preview_panel_img: return
        
        target_size, rot = self.s_size.get(), self.s_rot.get()
        bright, noise = self.s_bright.get(), self.s_noise.get()
        cx, cy = event.x, event.y
        
        self.placements.append({
            'cx': cx, 'cy': cy, 
            'target_size': target_size, 'rot': rot, 
            'bright': bright, 'noise': noise
        })

        img_rotated, _, _ = self.apply_transform(self.preview_panel_img, target_size, rot, bright, noise)
        paste_w, paste_h = img_rotated.size
        paste_x, paste_y = int(cx - paste_w / 2), int(cy - paste_h / 2)
        
        self.work_img.paste(img_rotated, (paste_x, paste_y), mask=img_rotated)
        self.redraw()
        self.lbl_status.config(text=f"Dodano pozycję. Razem obiektów: {len(self.placements)}")

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
            open(os.path.join(LBL_DIR, f"{empty_base_name}.txt"), "w").close() 

            # 2. Zapis wariantów paneli
            for p_idx, panel_path in enumerate(self.panel_images):
                panel_img = Image.open(panel_path).convert("RGBA")
                out_img = self.base_img.copy() 
                labels_obb = []
                
                for p in self.placements:
                    img_rot, bw, bh = self.apply_transform(
                        panel_img, p['target_size'], p['rot'], p['bright'], p['noise']
                    )
                    pw, ph = img_rot.size
                    px, py = int(p['cx'] - pw / 2), int(p['cy'] - ph / 2)

                    out_img.paste(img_rot, (px, py), mask=img_rot)
                    
                    corners = calculate_obb_corners(p['cx'], p['cy'], bw, bh, p['rot'])
                    norm_corners = normalize_obb(corners, out_img.width, out_img.height)
                    labels_obb.append((0, norm_corners))
                
                base_name = f"bg{self.current_bg_idx}_pnl{p_idx}_{uuid.uuid4().hex[:4]}"
                
                out_img.convert("RGB").save(os.path.join(IMG_DIR, f"{base_name}.jpg"))
                with open(os.path.join(LBL_DIR, f"{base_name}.txt"), "w") as f:
                    for cls, pts in labels_obb:
                        coords = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in pts])
                        f.write(f"{cls} {coords}\n")
                
                generate_dataset_variants(out_img, labels_obb, base_name)
                
            self.lbl_status.config(text="Zapisano pomyślnie. Ładowanie kolejnego tła...")
            
            # 3. Auto-przejście do następnego obrazka
            self.root.after(500, self.next_bg) 
            
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloObbApp(root)
    root.mainloop()