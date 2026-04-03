import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import random

# -----------------------------
# Utility
# -----------------------------

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def feather_mask(mask, ksize=7):
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)

def alpha_blend(fg, bg, mask):
    m = mask[..., None]
    return (fg * m + bg * (1 - m)).astype(np.uint8)

def blur_edges(fg, mask):
    blurred = cv2.GaussianBlur(fg, (5, 5), 0)
    edge = mask - cv2.erode(mask, np.ones((3,3),np.uint8), iterations=2)
    edge_3c = edge[..., None]
    return fg*(1-edge_3c) + blurred*edge_3c

def match_brightness_lab(fg, bg, mask):
    fg_lab = cv2.cvtColor(fg, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)

    m = mask.astype(bool)
    if np.sum(m)==0: return fg

    fg_L = fg_lab[...,0][m]
    bg_L = bg_lab[...,0][m]

    fg_lab[...,0] = (fg_lab[...,0]-fg_L.mean())*(bg_L.std()+1e-6)/(fg_L.std()+1e-6)+bg_L.mean()
    fg_lab = np.clip(fg_lab,0,255).astype(np.uint8)
    return cv2.cvtColor(fg_lab, cv2.COLOR_LAB2RGB)

def generate_shadow(mask):
    M = np.float32([[1,0,3],[0,1,3]])
    shadow = cv2.warpAffine(mask*255,M,(mask.shape[1],mask.shape[0]))
    shadow = cv2.GaussianBlur(shadow,(9,9),0)/255.0
    return shadow

def apply_shadow(bg, shadow):
    return (bg*(1-0.4*shadow[...,None])).astype(np.uint8)

def rotate(img, mask, angle):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2),angle,1)
    return (cv2.warpAffine(img,M,(w,h)),
            cv2.warpAffine(mask,M,(w,h)))

def get_poly(px,py,w,h,angle):
    cx,cy = px+w/2, py+h/2
    pts = np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
    R = np.array([[np.cos(np.deg2rad(angle)),-np.sin(np.deg2rad(angle))],
                  [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
    pts = pts @ R.T
    pts[:,0]+=cx; pts[:,1]+=cy
    return pts

# -----------------------------
# App
# -----------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.bg_paths = []
        self.panel_paths = []

        self.bg_idx = 0
        self.placements = []

        btn = tk.Frame(root)
        btn.pack()

        tk.Button(btn,text="Load BG Folder",command=self.load_bg).pack(side=tk.LEFT)
        tk.Button(btn,text="Load Panel Folder",command=self.load_panels).pack(side=tk.LEFT)
        tk.Button(btn,text="Save",command=self.save).pack(side=tk.LEFT)
        tk.Button(btn,text="Skip",command=self.next_bg).pack(side=tk.LEFT)

        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Button-3>", self.right_click)

    def load_bg(self):
        folder = 'C:/Users/admin/Desktop/inference_data/Inference_data/mck26/'
        self.bg_paths = [os.path.join(folder,f) for f in os.listdir(folder)]
        self.bg_idx = 0
        self.load_current_bg()

    def load_panels(self):
        folder = './pvs'
        self.panel_paths = [os.path.join(folder,f) for f in os.listdir(folder)]

    def load_current_bg(self):
        if self.bg_idx >= len(self.bg_paths):
            print("Done")
            return
        self.bg = load_image(self.bg_paths[self.bg_idx])
        self.display()

        self.placements = []

    def display(self):
        img = Image.fromarray(self.bg)
        self.tk = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0,0,anchor=tk.NW,image=self.tk)

    def left_click(self, e):
        count = random.randint(2,6)
        self.placements.append((e.x,e.y,count))
        self.preview()

    def right_click(self, e):
        self.placements.append((e.x,e.y,1))
        self.preview()

    def preview(self):
        img = self.bg.copy()
        for x,y,c in self.placements:
            for i in range(c):
                cv2.circle(img,(x+i*5,y+i*5),3,(255,0,0),-1)
        self.tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.create_image(0,0,anchor=tk.NW,image=self.tk)

    def generate(self, panel_img):
        out = self.bg.copy()
        polys = []

        for x,y,count in self.placements:
            for i in range(count):

                size = random.randint(30,50)
                px = x + i*size
                py = y

                panel = cv2.resize(panel_img,(size,size//2))
                mask = np.ones(panel.shape[:2],np.uint8)

                angle = random.uniform(-15,15)
                panel,mask = rotate(panel,mask,angle)

                h,w = panel.shape[:2]
                if py+h>=out.shape[0] or px+w>=out.shape[1]:
                    continue

                bg_patch = out[py:py+h,px:px+w]

                panel = match_brightness_lab(panel,bg_patch,mask)
                shadow = generate_shadow(mask)
                bg_patch = apply_shadow(bg_patch,shadow)

                panel = blur_edges(panel,mask)
                mask_soft = feather_mask(mask)

                blended = alpha_blend(panel,bg_patch,mask_soft)
                out[py:py+h,px:px+w] = blended

                polys.append(get_poly(px,py,w,h,angle))

        return out, polys

    def save(self):
        base_name = f"bg_{self.bg_idx:04d}"

        for p_idx, p_path in enumerate(self.panel_paths):
            panel_img = load_image(p_path)

            out, polys = self.generate(panel_img)

            img_name = f"{base_name}_panel_{p_idx:03d}.jpg"
            txt_name = img_name.replace(".jpg",".txt")

            cv2.imwrite(img_name, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

            h,w = out.shape[:2]
            with open(txt_name,"w") as f:
                for poly in polys:
                    norm = []
                    for x,y in poly:
                        norm.append(x/w); norm.append(y/h)
                    f.write("0 "+" ".join(map(str,norm))+"\n")

        print("Saved set for background", self.bg_idx)
        self.next_bg()

    def next_bg(self):
        self.bg_idx += 1
        self.load_current_bg()


# -----------------------------
# Run
# -----------------------------

root = tk.Tk()
app = App(root)
root.mainloop()