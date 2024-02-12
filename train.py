import argparse
from ultralytics import YOLO

def train_model(dataset_path, epochs, optimizer, learning_rate):
    # Load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt")
    
    # Train the model with the specified dataset, epochs, optimizer, and learning rate
    results = model.train(data=dataset_path, epochs=epochs, imgsz=640, optimizer=optimizer, lr0=learning_rate)
    
    # Optionally, print or save the training results
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model with specified dataset, number of epochs, and optional optimizer and learning rate.")
    
    parser.add_argument("dataset_path", type=str, help="Path to the dataset.yaml file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training. Default is 50.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer for training. Default is 'AdamW'.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer. Default is 1e-3.")
    
    args = parser.parse_args()
    
    # Call the training function with provided arguments
    train_model(args.dataset_path, args.epochs, args.optimizer, args.learning_rate)
