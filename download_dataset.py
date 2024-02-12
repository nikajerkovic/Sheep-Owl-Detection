import argparse
import fiftyone as fo
import fiftyone.zoo as foz

def download_dataset(export_dir, desired_splits, max_samples):
    label_field = "ground_truth"
    classes=["Owl", "Sheep"]
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        splits=desired_splits,
        label_types="detections",
        classes=classes,
        seed=51,
        max_samples=max_samples,
        shuffle=True,
        dataset_name="open-images-owl-sheep"
    )

    for split in desired_splits:
        split_view = dataset.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            split=split,
            classes=classes,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export a subset of the Open Images V7 dataset.")
    parser.add_argument(
        "--export_dir",
        type=str,
        required=True,
        help="Directory where the dataset will be exported."
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to download and export. Expects a list of splits separated by spaces. Default: 'train validation test'"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to download for each class. Default: 1000"
    )

    args = parser.parse_args()

    download_dataset(args.export_dir, args.splits, args.max_samples)
