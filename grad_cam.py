import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def visualize_gradcam(image_path):
    # 1. Load the pre-trained model
    # We use ResNet50 and set it to evaluation mode
    model = models.resnet50(pretrained=True)
    model.eval()

    # 2. Choose the target layer
    # For Grad-CAM, we typically choose the last convolutional layer.
    # In ResNet50, this is the last bottleneck block in layer4.
    target_layers = [model.layer4[-1]]

    # 3. Load and preprocess the image
    # Read the image using OpenCV (returns BGR), convert to RGB
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    
    # Resize image to match model's expected input (e.g., 224x224)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    
    # Preprocess the image (normalize using ImageNet mean and std)
    input_tensor = preprocess_image(rgb_img, 
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])

    # 4. Initialize the Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers)

    # 5. Define the target category
    # If set to None, Grad-CAM will default to the highest scoring class (the model's prediction).
    # To see what the model looks at for a specific class, pass its index: e.g., ClassifierOutputTarget(281) for 'tabby cat'.
    targets = None 

    # 6. Generate the heatmap
    # You can pass multiple images in the batch, but here we process just one [0]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 7. Overlay the heatmap onto the original image
    # show_cam_on_image applies a colormap (default is JET) to the grayscale heatmap and blends it.
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 8. Display the results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title('Grad-CAM Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Execution ---
# Replace 'path_to_your_image.jpg' with a real image file on your system
# visualize_gradcam('path_to_your_image.jpg')