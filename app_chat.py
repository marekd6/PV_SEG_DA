import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# -----------------------------
# Utility functions
# -----------------------------

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_for_display(img, max_size=800):
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    return cv2.resize(img, (int(w*scale), int(h*scale))), scale

# ---------- Brightness matching ----------
def match_brightness_lab(fg, bg, mask):
    fg_lab = cv2.cvtColor(fg, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)

    m = mask.astype(bool)
    if np.sum(m) == 0:
        return fg

    fg_L = fg_lab[..., 0][m]
    bg_L = bg_lab[..., 0][m]

    fg_mean, fg_std = fg_L.mean(), fg_L.std() + 1e-6
    bg_mean, bg_std = bg_L.mean(), bg_L.std() + 1e-6

    fg_lab[..., 0] = (fg_lab[..., 0] - fg_mean) * (bg_std / fg_std) + bg_mean

    fg_lab = np.clip(fg_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(fg_lab, cv2.COLOR_LAB2RGB)

# ---------- Shadow ----------
def generate_shadow(mask, offset=(3, 3), blur_ksize=9):
    h, w = mask.shape
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shadow = cv2.warpAffine(mask.astype(np.uint8)*255, M, (w, h))
    shadow = cv2.GaussianBlur(shadow, (blur_ksize, blur_ksize), 0)
    return shadow.astype(np.float32) / 255.0

def apply_shadow(bg, shadow, strength=0.4):
    shadow_3c = np.repeat(shadow[..., None], 3, axis=2)
    return (bg * (1 - strength * shadow_3c)).astype(np.uint8)

# ---------- Blending ----------
def feather_mask(mask, ksize=7):
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)

def alpha_blend(fg, bg, mask):
    m = mask[..., None]
    return (fg * m + bg * (1 - m)).astype(np.uint8)

def blur_edges(fg, mask, width=2):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=width)
    edge = mask - eroded

    blurred = cv2.GaussianBlur(fg, (5, 5), 0)

    edge_3c = edge[..., None]
    return fg * (1 - edge_3c) + blurred * edge_3c

# ---------- Rotation ----------
def rotate_image_and_mask(img, mask, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    img_r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    mask_r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    return img_r, mask_r

# ---------- Rotated polygon ----------
def get_rotated_polygon(px, py, w, h, angle):
    cx = px + w / 2
    cy = py + h / 2

    corners = np.array([
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2]
    ])

    theta = np.deg2rad(angle)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    rotated = corners @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    return rotated

# ---------- Debug draw ----------
def draw_polygon(img, poly):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(255,0,0), thickness=1)

# -----------------------------
# Main App
# -----------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Solar Panel Composer")

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.bg = None
        self.display_img = None
        self.scale = 1.0

        self.panel = None
        self.panel_mask = None

        self.polygons = []

        self.show_polygons = True  # toggle debug

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Load Background", command=self.load_bg).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Load Panel", command=self.load_panel).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Save", command=self.save).pack(side=tk.LEFT)

        self.canvas.bind("<Button-1>", self.on_click)

    def load_bg(self):
        path = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26/74865_1014763_N-34-50-C-c-4-4_0.jpg'
        self.bg = load_image(path)
        self.display_img, self.scale = resize_for_display(self.bg)

        self.tk_img = ImageTk.PhotoImage(Image.fromarray(self.display_img))
        self.canvas.config(width=self.display_img.shape[1], height=self.display_img.shape[0])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def load_panel(self):
        path = './pvs/p1.png'
        panel = load_image(path)
        mask = np.ones(panel.shape[:2], dtype=np.uint8)

        self.panel = panel
        self.panel_mask = mask

    def place_panel(self, x, y):
        size = 40
        grid = (3, 2)

        for i in range(grid[0]):
            for j in range(grid[1]):

                px = x + i * size
                py = y + j * size

                panel_resized = cv2.resize(self.panel, (size, size//2))
                mask_resized = cv2.resize(self.panel_mask, (size, size//2))

                angle = np.random.uniform(-10, 10)

                panel_r, mask_r = rotate_image_and_mask(panel_resized, mask_resized, angle)

                h, w = panel_r.shape[:2]

                if py+h >= self.bg.shape[0] or px+w >= self.bg.shape[1]:
                    continue

                bg_patch = self.bg[py:py+h, px:px+w]

                # Brightness
                panel_r = match_brightness_lab(panel_r, bg_patch, mask_r)

                # Shadow
                shadow = generate_shadow(mask_r, offset=(3, 3))
                bg_shadowed = apply_shadow(bg_patch, shadow)

                # Blending
                panel_r = blur_edges(panel_r, mask_r)
                soft_mask = feather_mask(mask_r, 7)

                blended = alpha_blend(panel_r, bg_shadowed, soft_mask)

                self.bg[py:py+h, px:px+w] = blended

                # --- Correct polygon ---
                poly = get_rotated_polygon(px, py, w, h, angle)
                self.polygons.append(poly)

        self.refresh_display()

    def refresh_display(self):
        disp = self.bg.copy()

        if self.show_polygons:
            for poly in self.polygons:
                draw_polygon(disp, poly)

        self.display_img, _ = resize_for_display(disp)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(self.display_img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_click(self, event):
        if self.panel is None:
            return

        x = int(event.x / self.scale)
        y = int(event.y / self.scale)

        self.place_panel(x, y)

    def save(self):
        img_path = "output.jpg"
        label_path = "output.txt"

        cv2.imwrite(img_path, cv2.cvtColor(self.bg, cv2.COLOR_RGB2BGR))

        h, w = self.bg.shape[:2]

        with open(label_path, "w") as f:
            for poly in self.polygons:
                norm = []
                for x, y in poly:
                    norm.append(x / w)
                    norm.append(y / h)

                line = "0 " + " ".join(f"{v:.6f}" for v in norm)
                f.write(line + "\n")

        print("Saved:", img_path, label_path)


# -----------------------------
# Run
# -----------------------------

root = tk.Tk()
app = App(root)
root.mainloop()