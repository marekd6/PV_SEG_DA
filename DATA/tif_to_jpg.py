import os
from PIL import Image

# Use the current directory where the script is located
directory = './dataset_yolo_seg35/both_r'

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a TIFF image
    if filename.lower().endswith(('.tif', '.tiff')):
        filepath = os.path.join(directory, filename)
        
        try:
            # Open the image
            with Image.open(filepath) as img:
                # Convert to RGB (Crucial, because JPGs do not support transparent backgrounds)
                rgb_im = img.convert('RGB')
                
                # Strip the old extension and add .jpg
                new_filename = os.path.splitext(filename)[0] + '.jpg'
                new_filepath = os.path.join(directory, new_filename)
                
                # Save the new file
                rgb_im.save(new_filepath, 'JPEG', quality=95)
                print(f"Successfully converted: {filename}  ->  {new_filename}")
                
        except Exception as e:
            print(f"Could not convert {filename}. Error: {e}")

print("Done!")
