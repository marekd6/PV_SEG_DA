'''
Docstring for preprocessing.img_preprocessing

splits orthophotos into small chunks
equal spatial size
equal px size (scaling)

I 26

from genAI adapted by MD
'''

import os
import time
import logging
from multiprocessing import Pool, cpu_count
from PIL import Image


Image.MAX_IMAGE_PIXELS = None
INPUT_FOLDER = './imagery125'
INPUT_FOLDER = '.'
OUTPUT_FOLDER = './mck26'
PREFIXES_SMALL = ('2019', '2020', '2021', '2023') 
SIZE_SMALL = 625
SIZE_LARGE = 3200
SIZE_LARGE = 625
TARGET_OUTPUT_SIZE = 625
QA = 92


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(processName)-15s] %(message)s',
    datefmt='%H:%M:%S'
)


def get_split_size(filename):
    if filename.startswith(PREFIXES_SMALL):
        return SIZE_SMALL
    return SIZE_LARGE


def process_single_image(file_path):
    """
    Worker function. Returns a result string or error message.
    """
    filename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    try:
        crop_size = get_split_size(filename)
        Image.MAX_IMAGE_PIXELS = None 
        start_job = time.time()
        
        with Image.open(file_path) as img:
            width, height = img.size
            tile_count = 0

            logging.info(f"STARTING FILE: {filename} | Mode: {crop_size}px split")
            
            for y in range(0, height, crop_size):
                for x in range(0, width, crop_size):
                    box = (x, y, min(x + crop_size, width), min(y + crop_size, height))
                    tile = img.crop(box)
                    
                    if crop_size == SIZE_LARGE:
                        tile = tile.resize((TARGET_OUTPUT_SIZE, TARGET_OUTPUT_SIZE), Image.Resampling.LANCZOS)
                        
                    out_name = f"{name_no_ext}_{tile_count}.jpg"
                    out_path = os.path.join(OUTPUT_FOLDER, out_name)
                    
                    tile.save(out_path, format="JPEG", quality=QA)
                    tile_count += 1
                    # logging.info(f"  -> Tile Saved: {out_name} (Source pos: x{x}, y{y})")
            
            elapsed = time.time() - start_job
            return f"[DONE] {filename:<30} | Size: {width}x{height} | Mode: {crop_size}px->{TARGET_OUTPUT_SIZE}px | Tiles: {tile_count} | Time: {elapsed:.2f}s"

    except Exception as e:
        return f"[ERROR] {filename:<30} | {str(e)}"


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    files_to_process = [
        os.path.join(INPUT_FOLDER, f) 
        for f in os.listdir(INPUT_FOLDER) 
        if f.lower().endswith(('.tif', '.tiff'))
    ]
    
    total_files = len(files_to_process)
    if not total_files:
        logging.warning("No TIF images found.")
        return

    logging.info(f"Starting parallel processing on {cpu_count()} cores for {total_files} files...")
    logging.info("-" * 80)

    start_time = time.time()

    with Pool(processes=2) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_image, files_to_process), 1):
            logging.info(f"({i}/{total_files}) {result}")

    duration = time.time() - start_time
    logging.info("-" * 80)
    logging.info(f"Batch completed in {duration:.2f} seconds.")


if __name__ == '__main__':
    main()
