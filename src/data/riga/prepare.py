import argparse
import json
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import re


class RIGADataPreparer:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = Path(raw_data_dir)
        
        # Pattern to match annotator masks: imageX-Y.tif where Y is annotator number
        self.mask_pattern = re.compile(r"(.+)-(\d+)\.tif$")
        # Pattern to match images: imageXprime.tif
        self.image_pattern = re.compile(r"(.+)prime\.tif$")

    def init_to_nnunet_raw_dataset_task(self, dataset_id, output_dir):
        """Initialize task to create nnUNet raw dataset.

        Args:
            dataset_id: Dataset ID for the nnUNet dataset
            output_dir: Output directory for nnUNet raw dataset
        """
        self.nnunet_dataset_id = dataset_id
        self.nnunet_output_dir = Path(output_dir)
        self.nnunet_output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_image_and_masks(self, source_dir, dataset_name):
        """
        Collect images and their corresponding multi-rater masks.
        
        Returns:
            dict: {image_id: {'image': Path, 'masks': [Path1, Path2, ...]}}
        """
        data = {}
        
        # Recursively find all .tif files
        for tif_file in source_dir.rglob("*.tif"):
            filename = tif_file.name
            
            # Check if it's an image
            image_match = self.image_pattern.match(filename)
            if image_match:
                image_id = f"{dataset_name}_{image_match.group(1)}"
                if image_id not in data:
                    data[image_id] = {'image': None, 'masks': []}
                data[image_id]['image'] = tif_file
                continue
            
            # Check if it's a mask
            mask_match = self.mask_pattern.match(filename)
            if mask_match:
                base_name = mask_match.group(1)
                annotator_id = int(mask_match.group(2))
                image_id = f"{dataset_name}_{base_name}"
                
                if image_id not in data:
                    data[image_id] = {'image': None, 'masks': []}
                data[image_id]['masks'].append((annotator_id, tif_file))
        
        # Sort masks by annotator ID for each image
        for image_id in data:
            data[image_id]['masks'] = [mask_path for _, mask_path in sorted(data[image_id]['masks'])]
        
        # Filter out entries without both image and masks
        data = {k: v for k, v in data.items() if v['image'] is not None and len(v['masks']) > 0}
        
        return data

    def _save_rgb_channels(self, image_path, output_dir, sample_id):
        """
        Load RGB image and save each channel separately.
        
        Returns:
            List of output paths for the three channels
        """
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Ensure it's RGB
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {img_array.shape}")
        
        output_paths = []
        for channel_idx in range(3):
            channel_data = img_array[:, :, channel_idx]
            channel_img = Image.fromarray(channel_data)
            output_path = output_dir / f"{sample_id}_{channel_idx:04d}.tif"
            channel_img.save(output_path)
            output_paths.append(output_path)
        
        return output_paths

    def _save_masks(self, mask_paths, output_dir, sample_id):
        """
        Save all annotator masks for a sample.
        
        Returns:
            List of output paths for the masks
        """
        output_paths = []
        for annotator_idx, mask_path in enumerate(mask_paths, start=1):
            mask_img = Image.open(mask_path)
            mask_array = np.array(mask_img)
            
            # Convert RGB mask to grayscale with color mappings
            # Bright red (255, 0, 0) -> 2, Less bright red (120, 0, 0) -> 1, else -> 0
            grayscale_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
            
            # Check for bright red (255, 0, 0)
            bright_red = np.all(mask_array == [255, 0, 0], axis=2)
            grayscale_mask[bright_red] = 2
            
            # Check for less bright red (120, 0, 0)
            less_bright_red = np.all(mask_array == [120, 0, 0], axis=2)
            grayscale_mask[less_bright_red] = 1
            
            # Save grayscale mask
            mask_img_gray = Image.fromarray(grayscale_mask)
            output_path = output_dir / f"{sample_id}_ann{annotator_idx}.tif"
            mask_img_gray.save(output_path)
            output_paths.append(output_path)
        
        return output_paths

    def _process_main_ds(self, main_ds_dirs, dataset_out_dir, dataset_entries):
        """
        Process main dataset (BinRushed + MESSIDOR) with 80:20 train/test split.
        """
        # Create output subdirectories
        imagesTr_outdir = dataset_out_dir / "imagesTr"
        labelsTr_outdir = dataset_out_dir / "labelsTr"
        imagesTs_outdir = dataset_out_dir / "imagesTs"
        labelsTs_outdir = dataset_out_dir / "labelsTs"
        imagesTr_outdir.mkdir(parents=True, exist_ok=True)
        labelsTr_outdir.mkdir(parents=True, exist_ok=True)
        imagesTs_outdir.mkdir(parents=True, exist_ok=True)
        labelsTs_outdir.mkdir(parents=True, exist_ok=True)
        
        # Collect all images and masks from main dataset directories
        all_data = {}
        for source_dir, dataset_name in main_ds_dirs:
            data = self._collect_image_and_masks(source_dir, dataset_name)
            all_data.update(data)
        
        print(f"Found {len(all_data)} images with masks in main dataset")
        
        # Split 80:20 train/test
        sample_ids = list(all_data.keys())
        random.shuffle(sample_ids)
        split_idx = int(len(sample_ids) * 0.8)
        train_ids = sample_ids[:split_idx]
        test_ids = sample_ids[split_idx:]
        
        # Process training set
        print(f"Processing {len(train_ids)} training samples...")
        for sample_id in tqdm(train_ids, desc="Processing training set"):
            sample_data = all_data[sample_id]
            # Save RGB channels
            self._save_rgb_channels(sample_data['image'], imagesTr_outdir, sample_id)
            # Save all annotator masks
            mask_paths = self._save_masks(sample_data['masks'], labelsTr_outdir, sample_id)
            
            # Add entry for each annotator to dataset_entries
            for ann_idx, mask_path in enumerate(mask_paths, start=1):
                dataset_entries[f"{sample_id}_ann{ann_idx}"] = {
                    "images": [
                        f"imagesTr/{sample_id}_0000.tif",
                        f"imagesTr/{sample_id}_0001.tif",
                        f"imagesTr/{sample_id}_0002.tif"
                    ],
                    "label": f"labelsTr/{sample_id}_ann{ann_idx}.tif"
                }
        
        # Process test set
        print(f"Processing {len(test_ids)} test samples...")
        for sample_id in tqdm(test_ids, desc="Processing test set"):
            sample_data = all_data[sample_id]
            # Save RGB channels
            self._save_rgb_channels(sample_data['image'], imagesTs_outdir, sample_id)
            # Save all annotator masks
            self._save_masks(sample_data['masks'], labelsTs_outdir, sample_id)
        
        return len(train_ids), len(test_ids)

    def _process_ood_ds(self, ood_ds_dirs, dataset_out_dir):
        """
        Process OOD dataset (Magrabia).
        """
        # Create output subdirectories
        imagesOodTs_outdir = dataset_out_dir / "imagesOodTs"
        labelsOodTs_outdir = dataset_out_dir / "labelsOodTs"
        imagesOodTs_outdir.mkdir(parents=True, exist_ok=True)
        labelsOodTs_outdir.mkdir(parents=True, exist_ok=True)
        
        # Collect all images and masks from OOD dataset directories
        all_data = {}
        for source_dir, dataset_name in ood_ds_dirs:
            data = self._collect_image_and_masks(source_dir, dataset_name)
            all_data.update(data)
        
        print(f"Found {len(all_data)} images with masks in OOD dataset")
        
        # Process all OOD samples
        print(f"Processing {len(all_data)} OOD samples...")
        for sample_id in tqdm(all_data.keys(), desc="Processing OOD set"):
            sample_data = all_data[sample_id]
            # Save RGB channels
            self._save_rgb_channels(sample_data['image'], imagesOodTs_outdir, sample_id)
            # Save all annotator masks
            self._save_masks(sample_data['masks'], labelsOodTs_outdir, sample_id)
        
        return len(all_data)

    def _create_dataset_json(self, dataset_out_dir, num_train, num_test, dataset_entries):
        """
        Create dataset.json file with dataset metadata.
        """
        dataset_json_content = {
            "name": "RIGA_multi_rater",
            "description": "RIGA dataset with multi-rater annotations. Groups BinRushed and MESSIDOR as main dataset (80:20 split), Magrabia as OOD test set.",
            "reference": "https://github.com/deep-retina/RIGA-dataset",
            "license": "Unknown",
            "release": "1.0",
            "file_ending": ".tif",
            "channel_names": {
                "0": "R",
                "1": "G",
                "2": "B"
            },
            "labels": {
                "background": 0,
                "disc": 1,
                "cup": 2
            },
            "numTraining": num_train,
            "numTest": num_test,
            "note": "Each image has multiple rater annotations (ann1, ann2, etc.).",
            "dataset": dataset_entries
        }
        
        with open(dataset_out_dir / "dataset.json", 'w') as f:
            json.dump(dataset_json_content, f, indent=4)

    def to_nnunet_raw_dataset(self):
        """
        Create nnUNet raw dataset from RIGA data with all rater annotations.
        
        BinRushed and MESSIDOR form the main dataset with 80:20 train-test split.
        Magrabia forms the OOD test dataset.
        All rater annotations per image are included.
        """
        print("Creating nnUNet raw dataset for RIGA with all rater annotations")
        
        # Create output directory
        dataset_out_dir = self.nnunet_output_dir / f"Dataset{self.nnunet_dataset_id}_RIGA_multi_rater"
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Define main dataset directories (BinRushed + MESSIDOR)
        main_ds_dirs = []
        
        # BinRushed subdirectories
        binrushed_base = self.raw_data_dir / "img_segmask_tif" / "BinRushedcorrected" / "BinRushed"
        for subdir in ["BinRushed1-Corrected", "BinRushed2", "BinRushed3", "BinRushed4"]:
            dir_path = binrushed_base / subdir
            if dir_path.exists():
                main_ds_dirs.append((dir_path, f"BinRushed_{subdir}"))
            else:
                print(f"Warning: {dir_path} not found")
        
        # MESSIDOR
        messidor_dir = self.raw_data_dir / "img_segmask_tif" / "MESSIDOR"
        if messidor_dir.exists():
            main_ds_dirs.append((messidor_dir, "MESSIDOR"))
        else:
            print(f"Warning: {messidor_dir} not found")
        
        # Define OOD dataset directories (Magrabia)
        ood_ds_dirs = []
        magrabia_base = self.raw_data_dir / "img_segmask_tif" / "Magrabia"
        for subdir in ["MagrabiaMale", "MagrabiFemale"]:
            dir_path = magrabia_base / subdir
            if dir_path.exists():
                ood_ds_dirs.append((dir_path, f"Magrabia_{subdir}"))
            else:
                print(f"Warning: {dir_path} not found")
        
        # Initialize dataset entries dictionary
        dataset_entries = {}
        
        # Process main dataset
        print("\n=== Processing Main Dataset ===")
        num_train, num_test = self._process_main_ds(main_ds_dirs, dataset_out_dir, dataset_entries)
        # num_train = 524
        # num_test = 131
        
        # Process OOD dataset
        print("\n=== Processing OOD Dataset ===")
        num_ood = self._process_ood_ds(ood_ds_dirs, dataset_out_dir)
        
        # Create dataset.json
        print("\n=== Creating dataset.json ===")
        self._create_dataset_json(dataset_out_dir, num_train, num_test, dataset_entries)
        
        print(f"\n=== Dataset Processing Complete ===")
        print(f"Training samples: {num_train}")
        print(f"Test samples: {num_test}")
        print(f"OOD samples: {num_ood}")
        print(f"Output directory: {dataset_out_dir}")


def main(args):
    preparer = RIGADataPreparer(raw_data_dir=args.raw_data_dir)

    if args.task == "to_nnunet_raw_dataset":
        preparer.init_to_nnunet_raw_dataset_task(
            dataset_id=args.nnunet_dataset_id,
            output_dir=args.nnunet_output_dir,
        )
        preparer.to_nnunet_raw_dataset()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RIGA dataset")
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
        help="Path to the raw RIGA dataset directory"
    )

    # args for 'to_nnunet_raw_dataset' task
    parser.add_argument(
        "--nnunet_dataset_id",
        type=str,
        required=False,
        help="Dataset ID for the nnUNet raw dataset (e.g., '004')",
    )
    parser.add_argument(
        "--nnunet_output_dir",
        type=str,
        required=False,
        help="Output directory for nnUNet raw dataset",
    )

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)

    main(args)
