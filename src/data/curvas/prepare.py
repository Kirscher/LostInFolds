import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from shutil import copy2
import SimpleITK as sitk


class CurvasDataPreparer:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = Path(raw_data_dir)
        
        # Load groups from Excel file
        self.groups_df = pd.read_csv(self.raw_data_dir / "CURVAS24_groups.csv")
        
        # Identify all patient directories across training_set, validation_set, testing_set
        self.patient_dirs = []
        for subset in ["training_set", "validation_set", "testing_set"]:
            subset_path = self.raw_data_dir / subset
            if subset_path.exists():
                self.patient_dirs.extend([d for d in subset_path.iterdir() if d.is_dir()])

    def init_to_nnunet_raw_dataset_task(self, dataset_id, output_dir):
        """Initialize task to create nnUNet raw dataset.

        Args:
            dataset_id: Dataset ID for the nnUNet dataset
            output_dir: Output directory for nnUNet raw dataset
        """
        self.nnunet_dataset_id = dataset_id
        self.nnunet_output_dir = Path(output_dir)
        self.nnunet_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_patient_group(self, patient_id):
        """Get the group (a, b, or c) for a given patient ID."""
        # Look up patient in groups dataframe
        patient_row = self.groups_df[self.groups_df.iloc[:, 0] == patient_id]
        if len(patient_row) == 0:
            print(f"Warning: Patient {patient_id} not found in groups file")
            return None
        return str(patient_row.iloc[0, 1]).strip("Group").lower()

    def to_nnunet_raw_dataset(self):
        """
        Create nnUNet raw dataset from CURVAS data with all rater annotations.

        Groups a and b form the main dataset with 80:20 train-test split.
        Group c forms the OOD test dataset.
        All three rater annotations per image are included.
        """
        print("Creating nnUNet raw dataset for CURVAS with all rater annotations")

        # Create nnUNet dataset directory
        nnunet_dataset_name = f"Dataset{self.nnunet_dataset_id}_CURVAS_all_raters"
        nnunet_dataset_dir = self.nnunet_output_dir / nnunet_dataset_name
        imagesTr_dir = nnunet_dataset_dir / "imagesTr"
        labelsTr_dir = nnunet_dataset_dir / "labelsTr"
        imagesTs_dir = nnunet_dataset_dir / "imagesTs"
        labelsTs_dir = nnunet_dataset_dir / "labelsTs"
        imagesOodTs_dir = nnunet_dataset_dir / "imagesOodTs"
        labelsOodTs_dir = nnunet_dataset_dir / "labelsOodTs"
        imagesTr_dir.mkdir(parents=True, exist_ok=True)
        labelsTr_dir.mkdir(parents=True, exist_ok=True)
        imagesTs_dir.mkdir(parents=True, exist_ok=True)
        labelsTs_dir.mkdir(parents=True, exist_ok=True)
        imagesOodTs_dir.mkdir(parents=True, exist_ok=True)
        labelsOodTs_dir.mkdir(parents=True, exist_ok=True)

        dataset_entries = {}

        # Separate patients by group
        group_a_patients = []
        group_b_patients = []
        group_c_patients = []

        for patient_dir in self.patient_dirs:
            patient_id = patient_dir.name
            group = self._get_patient_group(patient_id)
            
            if group == 'a':
                group_a_patients.append(patient_dir)
            elif group == 'b':
                group_b_patients.append(patient_dir)
            elif group == 'c':
                group_c_patients.append(patient_dir)

        # Combine groups a and b for main dataset
        main_dataset_patients = group_a_patients + group_b_patients
        
        # Shuffle and split 80:20
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(main_dataset_patients)
        split_idx = int(0.8 * len(main_dataset_patients))
        train_patients = main_dataset_patients[:split_idx]
        test_patients = main_dataset_patients[split_idx:]

        print(f"Train patients: {len(train_patients)}")
        print(f"Test patients: {len(test_patients)}")
        print(f"OOD patients (group c): {len(group_c_patients)}")

        # Process train patients
        for patient_dir in tqdm(train_patients, desc="Processing train patients"):
            self._process_patient(
                patient_dir, imagesTr_dir, labelsTr_dir, 
                dataset_entries, is_training=True
            )

        # Process test patients
        for patient_dir in tqdm(test_patients, desc="Processing test patients"):
            self._process_patient(
                patient_dir, imagesTs_dir, labelsTs_dir, 
                dataset_entries, is_training=False
            )

        # Process OOD patients (group c)
        for patient_dir in tqdm(group_c_patients, desc="Processing OOD patients"):
            self._process_patient(
                patient_dir, imagesOodTs_dir, labelsOodTs_dir, 
                dataset_entries, is_training=False
            )

        # Create dataset.json
        dataset_json = {
            "name": "CURVAS_all_raters",
            "description": "CURVAS dataset with groups a+b split 80:20 for train/test, group c as OOD test set. All three rater annotations included.",
            "reference": "https://curvas.grand-challenge.org/curvas-dataset/",
            "license": "CC BY-NC",
            "release": "1.0",
            "file_ending": ".nii.gz",
            "channel_names": {"0": "CT"},
            "labels": {
                "background": 0,
                "pancreas": 1,
                "kidney": 2,
                "liver": 3,
            },
            "numTraining": len(train_patients),
            "note": "Each image has three annotations, one per rater (ann1, ann2, ann3).",
            "dataset": dataset_entries,
        }

        # Save dataset.json
        with open(nnunet_dataset_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=4)

        print(f"nnUNet raw dataset created in {self.nnunet_output_dir}")
        print(f"Total training samples: {len(train_patients)} patients x 3 raters = {len(train_patients) * 3}")

    def _process_patient(self, patient_dir, images_dir, labels_dir, dataset_entries, is_training):
        """
        Process a single patient directory: copy image and all rater annotations.
        
        Args:
            patient_dir: Path to patient directory
            images_dir: Directory to save images
            labels_dir: Directory to save labels
            dataset_entries: Dictionary to add dataset entries (only for training)
            is_training: Whether this is a training sample (for dataset entries)
        """
        patient_id = patient_dir.name
        
        # Find image file
        image_file = patient_dir / "image.nii.gz"
        if not image_file.exists():
            print(f"Warning: Image not found for patient {patient_id}")
            return
        
        # Find all annotation files
        annotation_files = sorted(patient_dir.glob("annotation_*.nii.gz"))
        if len(annotation_files) != 3:
            print(f"Warning: Expected 3 annotations for patient {patient_id}, found {len(annotation_files)}")
            return
        
        # Copy image with nnUNet naming convention (single channel: _0000)
        image_output = images_dir / f"{patient_id}_0000.nii.gz"
        copy2(image_file, image_output)
        
        # Copy each annotation
        for ann_file in annotation_files:
            # Extract annotation number (1, 2, or 3)
            ann_name = ann_file.stem.replace(".nii", "")  # Remove .nii from .nii.gz
            ann_num = ann_name.split("_")[-1]  # Get the number after annotation_
            
            # Save label with rater identifier
            label_output = labels_dir / f"{patient_id}_ann{ann_num}.nii.gz"
            copy2(ann_file, label_output)
            
            # Add to dataset entries only for training samples
            if is_training:
                dataset_entries[f"{patient_id}_{ann_num}"] = {
                    "images": [f"{images_dir.name}/{patient_id}_0000.nii.gz"],
                    "label": f"{labels_dir.name}/{patient_id}_ann{ann_num}.nii.gz",
                }


def main(args):
    preparer = CurvasDataPreparer(raw_data_dir=args.raw_data_dir)

    if args.task == "to_nnunet_raw_dataset":
        preparer.init_to_nnunet_raw_dataset_task(
            dataset_id=args.nnunet_dataset_id,
            output_dir=args.nnunet_output_dir,
        )
        preparer.to_nnunet_raw_dataset()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CURVAS dataset")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to perform: 'to_nnunet_raw_dataset'",
    )

    parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        required=True, 
        help="Path to the raw CURVAS2024 directory"
    )

    # args for 'to_nnunet_raw_dataset' task
    parser.add_argument(
        "--nnunet_dataset_id",
        type=str,
        required=False,
        help="Dataset ID for the nnUNet raw dataset (e.g., '001')",
    )
    parser.add_argument(
        "--nnunet_output_dir",
        type=str,
        required=False,
        help="Output directory for nnUNet raw dataset",
    )

    args = parser.parse_args()

    main(args)
