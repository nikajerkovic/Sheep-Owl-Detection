import argparse
from ultralytics import YOLO
import os
from PIL import Image
import matplotlib.pyplot as plt

def get_full_image_path(image_name):
    # Define the base path for the images
    base_path = 'runs/detect/predict/'
    
    # Check if the image name is an absolute path or a relative path
    if os.path.isabs(image_name):
        # Absolute path - return as is
        return image_name
    else:
        # Relative path - prepend the base directory
        return os.path.join(base_path, image_name)

def load_model_and_predict(model_path, image_name):
    # Get the full image path
    image_path = get_full_image_path(image_name)
    
    # Load the model
    model_best = YOLO(model_path)
    
    # Perform prediction
    results = model_best.predict(source=image_path, save=True, boxes=True)
    
    # Print the results
    print(results)
    
    # Display the image with predictions using matplotlib
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load YOLO model and predict on an image.')
    parser.add_argument('model_path', type=str, help='Path to the YOLO model .pt file')
    parser.add_argument('image_name', type=str, help='Name or path to the image file')

    args = parser.parse_args()
    
    load_model_and_predict(args.model_path, args.image_name)
