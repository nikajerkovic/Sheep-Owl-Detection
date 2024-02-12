import argparse
import os
from PIL import Image
from ultralytics import YOLO  # Assuming you're using YOLO from the ultralytics package

def predict_from_folder(best_model_path, test_images_directory):
    # Load the best model
    best_model = YOLO(best_model_path)
    
    # Iterate over all images in the test directory
    for image_file in os.listdir(test_images_directory):
        # Check if file is an image
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the full path to the image file
            image_path = os.path.join(test_images_directory, image_file)

            # Load the image
            im = Image.open(image_path)

            # Perform prediction
            results = best_model.predict(source=im, save=True)  # Assuming 'save=True' saves the image with plotted boxes

            # Print results or handle them as needed
            print(results)

            # The saved image path might need to be adjusted depending on where 'best_model.predict' saves the images
            # Assuming images are saved in a specific directory, adjust the path as necessary
            saved_image_path = f'runs/detect/predict/{image_file}'
            # Optionally display or further process the saved images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform prediction on images from a specified folder.')
    parser.add_argument('best_model_path', type=str, help='Path to the best model weights file (best.pt).')
    parser.add_argument('test_images_directory', type=str, help='Path to the directory containing test images.')

    args = parser.parse_args()
    
    predict_from_folder(args.best_model_path, args.test_images_directory, )
