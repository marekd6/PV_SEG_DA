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

def feather_mask(mask, ksize=7):
    mask = mask.astype(np.float32)
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)

def alpha_blend(fg, bg, mask):
    m = mask[..., None]
    return (fg * m + bg * (1 - m)).astype(np.uint8)

def rotate_image_and_mask(img, mask, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    mask_r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return img_r, mask_r

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

        # simple rectangular mask
        mask = np.ones(panel.shape[:2], dtype=np.uint8)

        self.panel = panel
        self.panel_mask = mask

    def place_panel(self, x, y):
        size = 40  # base size
        grid = (3, 2)

        for i in range(grid[0]):
            for j in range(grid[1]):

                px = x + i * size
                py = y + j * size

                panel_resized = cv2.resize(self.panel, (size, size//2))
                mask_resized = cv2.resize(self.panel_mask, (size, size//2))

                # rotation
                angle = np.random.uniform(-10, 10)
                panel_r, mask_r = rotate_image_and_mask(panel_resized, mask_resized, angle)

                h, w = panel_r.shape[:2]

                if py+h >= self.bg.shape[0] or px+w >= self.bg.shape[1]:
                    continue

                bg_patch = self.bg[py:py+h, px:px+w]

                # simple brightness match
                panel_r = panel_r * 0.9 + bg_patch * 0.1
                panel_r = panel_r.astype(np.uint8)

                # blending
                mask_soft = feather_mask(mask_r, 5)
                blended = alpha_blend(panel_r, bg_patch, mask_soft)

                self.bg[py:py+h, px:px+w] = blended

                # polygon (YOLO format)
                poly = [
                    (px, py),
                    (px+w, py),
                    (px+w, py+h),
                    (px, py+h)
                ]
                self.polygons.append(poly)

        self.refresh_display()

    def refresh_display(self):
        self.display_img, _ = resize_for_display(self.bg)
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

                line = "0 " + " ".join(map(str, norm))
                f.write(line + "\n")

        print("Saved:", img_path, label_path)


# -----------------------------
# Run
# -----------------------------

root = tk.Tk()
app = App(root)
root.mainloop()