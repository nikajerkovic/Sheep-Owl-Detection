# Sheep and Owls Detection with YOLOv8
[![Made with YOLOv8](https://img.shields.io/badge/Made_with-YOLOv8-green)](https://docs.ultralytics.com/)


This repository contains the completed project for a hiring challenge. The task involved creating and training a specialized AI model to identify just two distinct objects, sheep and owls, within images. This objective was achieved utilizing the YOLOv8 (You Only Look Once) architecture, a cutting-edge framework in object detection. 

The solution provides a detailed Jupyter Notebook that walks through building and training an AI model. The code has been organized into scripts for added convenience, allowing for straightforward and efficient utilization. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zENtqvsL13vCZk2hRd5I2GWQ2_J9ECAv?usp=sharing)

## Installation and usage
In Command Prompt:

  1. Create Python virtual environment (optional but recommended)
     
     ```
     python -m venv <virtual-environment-name>
     env/Scripts/activate.bat
     ```
  2. Clone this repository
     
     ```
     git clone https://github.com/nikajerkovic/Sheep-Owl-Detection.git
     cd Sheep-Owl-Detection
     ```
  3. Install the requirements
     
     ```
     pip install -r requirements.txt
     ```
  4. Download the dataset (skip this if you want to use your test images)
     
     ```
     You can specify any combination of 'train', 'test', or 'validation' for the dataset splits.
     python download_dataset.py --export_dir "<path/to/export/directory>" --splits <split1> <split2> ... --max_samples <number_of_samples>
     ```
     or you can download the dataset from [here](https://drive.google.com/file/d/17hCNN3HpmSg63DIvscNlCNlb-wM1z20d/view?usp=sharing).  Once downloaded, unzip the file and place it in the Sheep-Owl-Detection directory.
     
  5. Train the model (optional)

     ```
     python train.py /path/to/your/dataset.yaml --epochs <number_of_epochs> --optimizer <optimizer_name> --learning_rate <learning_rate>
     ```
      
  6. Making predictions

      If you've trained the model, you can use the top-performing weights found at runs/detect/train/weights/best.pt. Alternatively, you can use the ones provided in this repository.
     
      To predict on a single image, use the following command:
      ```
      python predict.py /path/to/best.pt /path/to/image.jpg
      ```
      
      For processing an entire folder of images:
     
      ```
      python predict_from_folder.py /path/to/best.pt /path/to/test/images
      ```

## Results

The results from the most recent successful training session can be found [here](https://drive.google.com/file/d/1dMueyC95fHef9xHnQ-dEUaLpmrr7sRNh/view?usp=sharing). </br>
This directory also includes performance metrics on the test data located in detect/val. Of particular interest are the contents of detect/predict/predicted_and_true, where visual comparisons of the model predictions against the actual labels for all test images are provided. In the following examples from the specified folder, blue boxes represent the actual bounding boxes, while the red/peach boxes indicate the model predictions:

![alt text](https://github.com/nikajerkovic/Sheep-Owl-Detection/blob/main/image_for_readme.png)


## Resources

- [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html)
- [FiftyOne](https://docs.voxel51.com/)
- [Ultralytics for YOLOv8](https://docs.ultralytics.com/)


      
