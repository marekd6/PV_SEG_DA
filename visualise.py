import cv2
import os
import numpy as np

def visualize_masks(images_dir, labels_dir):
    # Get a sorted list of all image files in the directory
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    index = 0
    print("Controls: 'd' = Next image, 'a' = Previous image, 'q' = Quit")

    while index < len(image_files):
        img_name = image_files[index]
        img_path = os.path.join(images_dir, img_name)
        
        # Assume label file has the same name as the image, but with a .txt extension
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            index += 1
            continue
            
        h, w = img.shape[:2]
        overlay = img.copy() # Create a copy to draw translucent masks

        # Read and draw polygons if the label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: 
                    continue # Skip empty or invalid lines
                
                # class_id = int(parts[0]) # Extract class ID if you want to use different colors
                
                # Extract coordinates and reshape into pairs of (x, y)
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                
                # Unnormalize coordinates (multiply by image width and height)
                coords[:, 0] *= w
                coords[:, 1] *= h
                coords = coords.astype(np.int32)

                # Draw a filled polygon on the overlay
                # You can change the color (B, G, R) to match specific class IDs if needed
                cv2.fillPoly(overlay, [coords], color=(0, 255, 0)) 
                
                # Outline the polygon for better visibility
                cv2.polylines(overlay, [coords], isClosed=True, color=(0, 200, 0), thickness=2)

        # Blend the overlay with the original image to make the mask translucent (50% opacity)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Add image name text on top for easy tracking
        cv2.putText(img, f"[{index + 1}/{len(image_files)}] {img_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the image
        cv2.imshow('Segmentation Viewer', img)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'):   # 'd' key to move forward
            index += 1
        elif key == ord('a'): # 'a' key to move backward
            index = max(0, index - 1)
        elif key == ord('q'): # 'q' key to quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    IMAGES_FOLDER = 'dataset_yolo_seg8/images'
    LABELS_FOLDER = 'dataset_yolo_seg8/labels'
    
    visualize_masks(IMAGES_FOLDER, LABELS_FOLDER)