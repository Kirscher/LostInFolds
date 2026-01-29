import argparse
import json
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import SimpleITK as sitk


class GleasonXAIDataPreparer:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = Path(raw_data_dir)

        self.explanation2gleasongrade_mapping = (
            self._load_gleason_explanation_mappings()
        )

        self.gleasongrade2seglabel_mapping = {
            "0": 0,
            "3": 1,
            "4": 2,
            "5": 3,
        }

        # get list of all raw image files
        # TissueArray
        self.tissue_array_raw_imgs = glob(
            str(
                self.raw_data_dir
                / "27301845"
                / "tissuearray_com_data"
                / "tissuemicroarray_com_data"
                / "*.jpg"
            )
        )
        # Harvard Dataverse
        self.harvard_dataverse_raw_imgs = glob(
            str(self.raw_data_dir / "dataverse_files" / "*" / "*.jpg")
        )
        self.harvard_dataverse_raw_imgs = [
            img for img in self.harvard_dataverse_raw_imgs if "mask" not in img
        ]
        # Gleason2019
        self.gleason2019_raw_imgs = glob(
            str(
                self.raw_data_dir
                / ".."
                / "Gleason2019"
                / "raw_grandchallenge"
                / "Train_imgs"
                / "*.jpg"
            )
        )
        self.gleason2019_raw_imgs += glob(
            str(
                self.raw_data_dir
                / ".."
                / "Gleason2019"
                / "raw_grandchallenge"
                / "Test_imgs"
                / "*.jpg"
            )
        )

    def _load_gleason_explanation_mappings(self):
        gleasongrade2explanation_mapping = json.load(
            open(self.raw_data_dir / "27301845" / "label_remapping.json")
        )["hierarchy"]
        explanation2gleasongrade_mapping = {}
        for grade, categories in gleasongrade2explanation_mapping.items():
            for category, explanations in categories.items():
                # Map category name to grade
                explanation2gleasongrade_mapping[category] = int(grade)
                # Map each detailed explanation to grade
                for expl in explanations:
                    explanation2gleasongrade_mapping[expl] = int(grade)
        return explanation2gleasongrade_mapping

    def init_generate_labels_task(self, output_dir):
        self.generate_labels_output_dir = Path(output_dir)
        self.generate_labels_output_dir.mkdir(parents=True, exist_ok=True)
        # also create subdirectories for each sub dataset
        for datasource in ["tissue_array", "harvard_dataverse", "gleason2019"]:
            (self.generate_labels_output_dir / datasource / "all_raters").mkdir(
                parents=True, exist_ok=True
            )

    def init_convert_images_task(self, output_dir, generate_labels_output_dir, convert_images_mode="jpg_to_png"):
        self.convert_img_mode = convert_images_mode
        self.convert_output_dir = Path(output_dir)
        self.convert_output_dir.mkdir(parents=True, exist_ok=True)
        self.generate_labels_output_dir = Path(generate_labels_output_dir)

    def init_to_nnunet_raw_dataset_task(
        self, dataset_id, labels_input_dir, images_input_dir, main_ds, ood_ds, output_dir
    ):
        """Initialize task to create nnUNet raw datasets.

        Args:
            dataset_id: Dataset ID for the nnUNet dataset
            labels_input_dir: Directory containing generated labels (output from generate_labels)
            images_input_dir: Directory containing converted images (output from convert_images)
            main_ds: List of subdatasets composing the main dataset
            ood_ds: List of subdatasets composing the OOD dataset
            output_dir: Output directory for nnUNet raw datasets
        """
        self.nnunet_dataset_id = dataset_id
        self.nnunet_labels_input_dir = Path(labels_input_dir)
        self.nnunet_images_input_dir = Path(images_input_dir)
        self.main_ds = main_ds
        self.ood_ds = ood_ds
        self.nnunet_output_dir = Path(output_dir)
        self.nnunet_output_dir.mkdir(parents=True, exist_ok=True)

    def _load_image(self, tma_identifier):
        # search for the image in all raw image lists
        for img_list in [
            self.tissue_array_raw_imgs,
            self.harvard_dataverse_raw_imgs,
            self.gleason2019_raw_imgs,
        ]:
            for img_path in img_list:
                if tma_identifier in img_path:
                    datasource = ""
                    if "tissuemicroarray_com_data" in img_path:
                        datasource = "tissue_array"
                    elif "dataverse_files" in img_path:
                        datasource = "harvard_dataverse"
                    elif "Gleason2019" in img_path:
                        datasource = "gleason2019"
                    # load and return image
                    return Image.open(img_path), datasource

        raise FileNotFoundError(
            f"Image with TMA_identifier {tma_identifier} not found."
        )

    def _save_label_mask(
        self, label_mask, tma_identifier, datasource, rater_name, expected_size=None
    ):
        """Save label mask with optional size validation.

        Args:
            label_mask: The label mask array
            tma_identifier: TMA identifier for the image
            datasource: Source dataset name
            rater_name: Name of the rater
            expected_size: Optional (H, W) tuple to validate mask size
        """
        # Validate mask size if expected size is provided
        if expected_size is not None:
            actual_size = label_mask.shape[:2]  # (H, W)
            if actual_size != expected_size:
                raise ValueError(
                    f"Mask size mismatch for {tma_identifier}: "
                    f"expected {expected_size}, got {actual_size}. "
                    f"Ensure image and mask are generated with same dimensions."
                )

        # save label mask with rater name in filename
        output_path = (
            self.generate_labels_output_dir
            / datasource
            / "all_raters"
            / f"{tma_identifier}_rater_{rater_name}_mask.png"
        )
        Image.fromarray(label_mask).save(output_path)
        print(f"Saved label mask to {output_path}")

    def _generate_rater_label(self, tma_df, rater_name, img_size):
        """Generate label mask for a single rater.

        Args:
            tma_df: DataFrame containing annotations for a single TMA
            rater_name: Name of the rater
            img_size: Size of the image as (H, W)

        Returns:
            Label mask array for this rater
        """
        rater_df = tma_df[tma_df["annotator"] == rater_name]

        label_mask = np.zeros(img_size, dtype=np.uint8)
        for exp, coords in zip(
            rater_df["explanations"],
            rater_df["coords"],
        ):
            coords = np.array(eval(coords))
            new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :])
            label_slice = np.zeros(list(img_size), dtype=np.int8)

            # cv2.fillPoly expects (W,H) coordinates
            cv2.fillPoly(label_slice, [new_coords.T], color=1)

            gleason_grade = self.explanation2gleasongrade_mapping[exp]
            seg_label = self.gleasongrade2seglabel_mapping[str(gleason_grade)]
            label_mask[label_slice > 0] = seg_label

        return label_mask

    def generate_labels(self):
        """Generate labels for all raters for each image.
        
        For each image, creates a separate label mask for each rater.
        Masks are saved with rater identifiers in filenames.
        """
        print("Generating labels for all raters")

        # load final_filtered_explanations_df.csv from raw_data_dir
        df = pd.read_csv(
            f"{self.raw_data_dir}/27301845/final_filtered_explanations_df.csv"
        )

        # get unique images ("TMA_identifier") and image-rater pairs
        all_tmas = df["TMA_identifier"].unique()
        all_tmas_raters_pairs = df[["TMA_identifier", "annotator"]].drop_duplicates()

        for tma in tqdm(all_tmas, desc="Processing TMAs"):
            # get all tma_rater_pair for this tma
            tma_raters_pairs = all_tmas_raters_pairs[
                all_tmas_raters_pairs["TMA_identifier"] == tma
            ]

            # get df for this tma
            tma_df = df[df["TMA_identifier"] == tma]

            # load image for reference
            img, datasource = self._load_image(tma_df.iloc[0]["TMA_identifier"])
            img_size = np.array(img.size)[::-1]  # (H, W)

            # Generate label mask for each rater
            for _, row in tma_raters_pairs.iterrows():
                rater = row["annotator"]
                label_mask = self._generate_rater_label(tma_df, rater, img_size)

                # save label mask with size validation
                self._save_label_mask(
                    label_mask,
                    tma_df.iloc[0]["TMA_identifier"],
                    datasource,
                    rater,
                    expected_size=tuple(img_size),
                )

    def convert_images(self):
        """Convert all images to new format given convert_images_mode."""
        print(f"Converting images from {self.convert_img_mode} format...")

        # Collect all image lists
        all_img_lists = [
            (self.tissue_array_raw_imgs, "tissue_array"),
            (self.harvard_dataverse_raw_imgs, "harvard_dataverse"),
            (self.gleason2019_raw_imgs, "gleason2019"),
        ]

        # load annotation table to only convert images that have annotations
        df = pd.read_csv(
            f"{self.raw_data_dir}/27301845/final_filtered_explanations_df.csv"
        )

        total_images = sum(len(img_list) for img_list, _ in all_img_lists)

        for img_list, datasource in all_img_lists:
            # Create output subdirectory for this datasource
            output_subdir = self.convert_output_dir / datasource
            output_subdir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(img_list, desc=f"Converting {datasource} images"):
                # check if image has annotations
                tma_identifier = Path(img_path).stem
                if tma_identifier not in df["TMA_identifier"].values:
                    continue

                # Load image
                img = Image.open(img_path)
                img_size = img.size  # (W, H)

                # Load a reference mask to check size match
                mask_dir =  self.generate_labels_output_dir / datasource / "all_raters"
                mask_files = list(mask_dir.glob(f"{tma_identifier}_rater_*_mask.png"))
                
                if mask_files:
                    mask = Image.open(mask_files[0])
                    mask_size = mask.size  # (W, H)
                    if img_size != mask_size:
                        # resize image to mask size
                        print(
                            f"Resizing image {tma_identifier} from size {img_size} to mask size {mask_size}"
                        )
                        img = img.resize(mask_size, Image.LANCZOS)
                        img_size = img.size  # update size after resize
                    assert (
                        img_size == mask_size
                    ), f"Size mismatch for {tma_identifier}: image {img_size} vs mask {mask_size}"

                # Get filename without extension
                img_filename = Path(img_path).stem

                if self.convert_img_mode == "jpg_to_png":
                    # Save as PNG
                    output_path = output_subdir / f"{img_filename}.png"
                    img.save(output_path, "PNG")
                else:
                    raise ValueError(
                        f"Unknown convert_images_mode: {self.convert_img_mode}"
                    )

        print(f"Converted {total_images} images to PNG format.")

    def to_nnunet_raw_dataset(self):
        """
        Create nnUNet raw datasets from all rater labels and images.

        Subdatasets in main_ds are split with a 80:20 split (default) into train and test sets.
        OOD datasets are copied entirely to ood-test set.
        Images of respective datasets are loaded, separated into RGB channels, and saved accordingly.
        Labels are saved with rater identifiers in filenames.
        dataset.json is created for whole dataset.

        Args:
            main_ds: list of nnunet_rawraw directories for the main dataset
            ood_ds: list of nnunet_rawraw directories for the OOD dataset
        """
        print("Creating nnUNet raw datasets with all rater labels")

       
        # Create nnUNet dataset directory
        nnunet_dataset_name = f"Dataset{self.nnunet_dataset_id}_GleasonXAI_all_raters"
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

        # process main dataset
        # split main_ds samples 80:20 to train and test sets
        sample_ids_main_ds = []
        for datasource in self.main_ds:
            # Get images for this datasource
            images_subdir = self.nnunet_images_input_dir / datasource
            if not images_subdir.exists():
                print(f"Warning: Images directory not found: {images_subdir}")
                image_files = []
            else:
                image_files = list(images_subdir.glob("*.png"))

            for img_file in image_files:
                tma_identifier = img_file.stem
                sample_ids_main_ds.append((tma_identifier, datasource))
        np.random.shuffle(sample_ids_main_ds)
        split_idx = int(0.8 * len(sample_ids_main_ds))
        train_sample_ids_main_ds = sample_ids_main_ds[:split_idx]
        test_sample_ids_main_ds = sample_ids_main_ds[split_idx:]

        # process main_ds' train and test samples
        for set in ["train", "test"]:
            set_ids_main_ds = train_sample_ids_main_ds if set == "train" else test_sample_ids_main_ds
            imgs_dir = imagesTr_dir if set == "train" else imagesTs_dir
            labels_dir = labelsTr_dir if set == "train" else labelsTs_dir
            for sample_id in tqdm(set_ids_main_ds, desc="Processing main_ds train samples"):
                tma_identifier, datasource = sample_id

                # Get images for this sample
                img_file = self.nnunet_images_input_dir / datasource / f"{tma_identifier}.png"
                assert img_file.exists(), f"Image file not found: {img_file}"

                # Get all rater labels for this sample
                labels_subdir = (
                    self.nnunet_labels_input_dir / datasource / "all_raters"
                )
                rater_label_files = sorted(
                    labels_subdir.glob(f"{tma_identifier}_rater_*_mask.png")
                )
                assert rater_label_files, f"Warning: No rater labels found for image {tma_identifier}, skipping"

                # Load image and convert to RGB if needed
                img = Image.open(img_file).convert("RGB")
                img_array = np.array(img)  # (H, W, 3) with RGB channels

                # Save image channels with nnUNet naming convention (only once per image)
                # Channel 0 (R), 1 (G), 2 (B)
                r_channel = img_array[:, :, 0]
                g_channel = img_array[:, :, 1]
                b_channel = img_array[:, :, 2]

                for channel_idx, channel_data in enumerate(
                    [r_channel, g_channel, b_channel]
                ):
                    channel_img = Image.fromarray(channel_data, mode="L")
                    channel_filename = f"{tma_identifier}_{channel_idx:04d}.png"
                    channel_img.save(imgs_dir / channel_filename)

                # Save all rater labels
                for rater_label_file in rater_label_files:
                    # Extract rater name from filename
                    # Format: {tma_identifier}_rater_{rater_name}_mask.png
                    parts = rater_label_file.stem.split("_rater_")
                    rater_name = parts[1].replace("_mask", "")

                    # Load label
                    label_img = Image.open(rater_label_file)
                    label_array = np.array(label_img)  # (H, W)

                    # Verify sizes match
                    if img_array.shape[:2] != label_array.shape[:2]:
                        raise ValueError(
                            f"Image-label size mismatch for {tma_identifier} (rater {rater_name}): "
                            f"image {img_array.shape[:2]} vs label {label_array.shape[:2]}. "
                        )
                    
                    # Save label with rater identifier
                    label_filename = f"{tma_identifier}_rater_{rater_name}.png"
                    label_img.save(labels_dir / label_filename)

                    if set == "train":
                        dataset_entries[Path(label_filename).stem] = {
                            "images": [
                                f"{imgs_dir.name}/{tma_identifier}_{idx:04d}.png"
                                for idx in range(3)
                            ],
                            "label": f"{labels_dir.name}/{label_filename}",
                        }

        # process OOD dataset
        for datasource in self.ood_ds:
            # Get images for this datasource
            images_subdir = self.nnunet_images_input_dir / datasource
            if not images_subdir.exists():
                print(f"Warning: Images directory not found: {images_subdir}")
                image_files = []
            else:
                image_files = list(images_subdir.glob("*.png"))

            for img_file in tqdm(image_files, desc=f"Processing OOD datasource {datasource}"):
                tma_identifier = img_file.stem

                # Get all rater labels for this image
                labels_subdir = (
                    self.nnunet_labels_input_dir / datasource / "all_raters"
                )
                rater_label_files = sorted(
                    labels_subdir.glob(f"{tma_identifier}_rater_*_mask.png")
                )
                
                if not rater_label_files:
                    print(
                        f"Warning: No rater labels found for image {tma_identifier}, skipping"
                    )
                    continue

                # Load image and convert to RGB if needed
                img = Image.open(img_file).convert("RGB")
                img_array = np.array(img)  # (H, W, 3) with RGB channels

                # Save image channels with nnUNet naming convention (only once per image)
                # Channel 0 (R), 1 (G), 2 (B)
                r_channel = img_array[:, :, 0]
                g_channel = img_array[:, :, 1]
                b_channel = img_array[:, :, 2]

                for channel_idx, channel_data in enumerate(
                    [r_channel, g_channel, b_channel]
                ):
                    channel_img = Image.fromarray(channel_data, mode="L")
                    channel_filename = f"{tma_identifier}_{channel_idx:04d}.png"
                    channel_img.save(imagesOodTs_dir / channel_filename)

                # Save all rater labels
                for rater_label_file in rater_label_files:
                    # Extract rater name from filename
                    # Format: {tma_identifier}_rater_{rater_name}_mask.png
                    parts = rater_label_file.stem.split("_rater_")
                    rater_name = parts[1].replace("_mask", "")

                    # Load label
                    label_img = Image.open(rater_label_file)
                    label_array = np.array(label_img)  # (H, W)

                    # Verify sizes match
                    if img_array.shape[:2] != label_array.shape[:2]:
                        raise ValueError(
                            f"Image-label size mismatch for {tma_identifier} (rater {rater_name}): "
                            f"image {img_array.shape[:2]} vs label {label_array.shape[:2]}. "
                        )
                    # Save label with rater identifier
                    label_filename = f"{tma_identifier}_rater_{rater_name}.png"
                    label_img.save(labelsOodTs_dir / label_filename)

                    # do not add OOD samples to dataset entries

        # Create dataset.json
        dataset_json = {
            "name": f"GleasonXAI_all_raters",
            "description": f"GleasonXAI dataset with 80:20 splitted main dataset {self.main_ds} \
                 and OoD dataset {self.ood_ds}, all with labels from all raters.",
            "reference": "GleasonXAI",
            "license": "CC-BY-4.0",
            "release": "1.0",
            "file_ending": ".png",
            "channel_names": {"0": "R", "1": "G", "2": "B"},
            "labels": {
                "background": 0,
                "gleason_3": 1,
                "gleason_4": 2,
                "gleason_5": 3,
            },
            "numTraining": len(dataset_entries),
            "note": "Each image has multiple labels, one per rater, saved with rater identifiers in filenames.",
            "dataset": dataset_entries,

        }

        # Save dataset.json
        with open(nnunet_dataset_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=4)

        print(f"nnUNet raw datasets created in {self.nnunet_output_dir}")


def main(args):
    preparer = GleasonXAIDataPreparer(raw_data_dir=args.raw_data_dir)

    if args.task == "generate_labels":
        preparer.init_generate_labels_task(
            output_dir=args.generate_labels_output_dir,
        )
        preparer.generate_labels()
    elif args.task == "convert_images":
        preparer.init_convert_images_task(
            convert_images_mode=args.convert_images_mode,
            output_dir=args.convert_images_output_dir,
            generate_labels_output_dir=args.generate_labels_output_dir,
        )
        preparer.convert_images()
    elif args.task == "to_nnunet_raw_dataset":
        preparer.init_to_nnunet_raw_dataset_task(
            dataset_id=args.nnunet_dataset_id,
            labels_input_dir=args.nnunet_labels_input_dir,
            images_input_dir=args.nnunet_images_input_dir,
            main_ds=args.main_ds,
            ood_ds=args.ood_ds,
            output_dir=args.nnunet_output_dir,
        )
        preparer.to_nnunet_raw_dataset()
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Gleason XAI dataset")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to perform: 'generate_labels', 'convert_images', 'to_nnunet_raw_dataset'",
    )

    parser.add_argument(
        "--raw_data_dir", type=str, required=True, help="Path to the raw data directory"
    )

    # args for 'generate_labels' task
    parser.add_argument(
        "--generate_labels_output_dir",
        type=str,
        required=False,
        help="Output directory for generated labels",
    )

    # args for 'convert_images' task
    parser.add_argument(
        "--convert_images_mode",
        type=str,
        required=False,
        help="Mode for image conversion: 'jpg_to_png', ...",
    )
    parser.add_argument(
        "--convert_images_output_dir",
        type=str,
        required=False,
        help="Output directory for converted images.",
    )

    # args for 'to_nnunet_raw_dataset' task
    parser.add_argument(
        "--nnunet_dataset_id",
        type=str,
        required=False,
        help="Dataset ID for the nnUNet raw dataset (e.g., '003')",
    )
    parser.add_argument(
        "--nnunet_labels_input_dir",
        type=str,
        required=False,
        help="Input directory containing generated labels (output from generate_labels task)",
    )
    parser.add_argument(
        "--nnunet_images_input_dir",
        type=str,
        required=False,
        help="Input directory containing converted images (output from convert_images task)",
    )
    parser.add_argument(
        "--main_ds",
        type=str,
        nargs="+",
        required=False,
        help="List of subdatasets composing the main dataset",
    )
    parser.add_argument(
        "--ood_ds",
        type=str,
        nargs="+",
        required=False,
        help="List of subdatasets composing the OOD dataset",
    )
    parser.add_argument(
        "--nnunet_output_dir",
        type=str,
        required=False,
        help="Output directory for nnUNet raw datasets",
    )

    args = parser.parse_args()

    main(args)