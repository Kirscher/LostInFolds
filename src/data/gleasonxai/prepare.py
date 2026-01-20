from pathlib import Path
from shutil import copy2
import random
from tqdm import tqdm
import json

def process_ood_ds(nnunet_rawraw_dirs, dataset_out_dir):
    # create output subdirectories
    imagesTs_outdir = dataset_out_dir / "imagesOodTs"
    labelsTs_outdir = dataset_out_dir / "labelsOodTs"
    imagesTs_outdir.mkdir(parents=True, exist_ok=True)
    labelsTs_outdir.mkdir(parents=True, exist_ok=True)

    # Collect all image and label files
    all_img_files = []
    all_label_files = []

    for nnunet_rawraw_dir in nnunet_rawraw_dirs:
        imagesTr_srcdir = nnunet_rawraw_dir / "imagesTr"
        labelsTr_srcdir = nnunet_rawraw_dir / "labelsTr"
        all_img_files.extend(imagesTr_srcdir.iterdir())
        all_label_files.extend(labelsTr_srcdir.iterdir())

    # Copy all files to test set
    for img_file in tqdm(all_img_files, desc="Copying OOD test images"):
        copy2(img_file, imagesTs_outdir / img_file.name)
    for label_file in tqdm(all_label_files, desc="Copying OOD test labels"):
        copy2(label_file, labelsTs_outdir / label_file.name)

def process_main_ds(nnunet_rawraw_dirs, dataset_out_dir):
    # create output subdirectories
    imagesTr_outdir = dataset_out_dir / "imagesTr"
    labelsTr_outdir = dataset_out_dir / "labelsTr"
    imagesTs_outdir = dataset_out_dir / "imagesTs"
    labelsTs_outdir = dataset_out_dir / "labelsTs"
    imagesTr_outdir.mkdir(parents=True, exist_ok=True)
    labelsTr_outdir.mkdir(parents=True, exist_ok=True)
    imagesTs_outdir.mkdir(parents=True, exist_ok=True)
    labelsTs_outdir.mkdir(parents=True, exist_ok=True)

    # Collect all image and label files
    all_img_files = []
    all_label_files = []

    for nnunet_rawraw_dir in nnunet_rawraw_dirs:
        imagesTr_srcdir = nnunet_rawraw_dir / "imagesTr"
        labelsTr_srcdir = nnunet_rawraw_dir / "labelsTr"
        all_img_files.extend(imagesTr_srcdir.iterdir())
        all_label_files.extend(labelsTr_srcdir.iterdir())

    # Create filename mapping for labels
    label_map = {label_file.stem: label_file for label_file in all_label_files}

    # Group images by sample (remove channel suffix)
    sample_groups = {}
    for img_file in all_img_files:
        # Extract base name without channel suffix (e.g., "sample_0" from "sample_0_0000.nii.gz")
        parts = img_file.stem.rsplit('_', 1)
        base_name = parts[0] if len(parts) > 1 else img_file.stem
        if base_name not in sample_groups:
            sample_groups[base_name] = []
        sample_groups[base_name].append(img_file)

    # Shuffle and split 80-20 by sample groups
    sample_list = list(sample_groups.keys())
    random.shuffle(sample_list)
    split_idx = int(len(sample_list) * 0.8)
    train_samples = sample_list[:split_idx]
    test_samples = sample_list[split_idx:]

    # Collect all channel images and corresponding labels for train/test
    train_img_files = set([img for sample in train_samples for img in sample_groups[sample]])
    test_img_files = set([img for sample in test_samples for img in sample_groups[sample]])
    train_label_files = set([label_map[parts[0]] for img in train_img_files if (parts := img.stem.rsplit('_', 1))[0] in label_map])
    test_label_files = set([label_map[parts[0]] for img in test_img_files if (parts := img.stem.rsplit('_', 1))[0] in label_map])

    # Copy files
    for img_file in tqdm(train_img_files, desc="Copying train images"):
        copy2(img_file, imagesTr_outdir / img_file.name)
    for label_file in tqdm(train_label_files, desc="Copying train labels"):
        copy2(label_file, labelsTr_outdir / label_file.name)
    for img_file in tqdm(test_img_files, desc="Copying test images"):
        copy2(img_file, imagesTs_outdir / img_file.name)
    for label_file in tqdm(test_label_files, desc="Copying test labels"):
        copy2(label_file, labelsTs_outdir / label_file.name)

    # created dataset.json file
    dataset_json_content = {
        "name": "GleasonXAI_consensus_staple",
        "description": "GleasonXAI dataset with consensus staple labels",
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "gleason_3": 1,
            "gleason_4": 2,
            "gleason_5": 3
        },
        "numTraining": len(train_label_files),
        "numTest": len(test_label_files),
    }
    with open(dataset_out_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json_content, f, indent=4)
    
def main(nnunet_rawraw_dirs, nnunet_raw_outdir, dataset_id):
    # create output directory if it doesn't exist
    nnunet_raw_outdir.mkdir(parents=True, exist_ok=True)
    dataset_out_dir = nnunet_raw_outdir / f"Dataset{dataset_id}_GleasonXAI_consensus_staple"
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    process_main_ds(nnunet_rawraw_dirs["main_ds"], dataset_out_dir)

    process_ood_ds(nnunet_rawraw_dirs["ood_ds"], dataset_out_dir)

if __name__=="__main__":
    nnunet_rawraw_dir1 = Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/GleasonXAI/nnUNet_raw/Dataset430_GleasonXAI_consensus_staple_client0")
    nnunet_rawraw_dir2 = Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/GleasonXAI/nnUNet_raw/Dataset431_GleasonXAI_consensus_staple_client1")
    nnunet_rawraw_dir3_ood = Path("/home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/GleasonXAI/nnUNet_raw/Dataset432_GleasonXAI_consensus_staple_client2")
    nnunet_rawraw_dirs = {
        "main_ds": [nnunet_rawraw_dir1, nnunet_rawraw_dir2],
        "ood_ds": [nnunet_rawraw_dir3_ood]
    }

    nnunet_raw_outdir = Path("/home/m391k/E132-Projekte/Projects/2026_Kirscher_LostInFolds/data/GleasonXAI")
    dataset_id = "001"

    main(nnunet_rawraw_dirs, nnunet_raw_outdir, dataset_id)