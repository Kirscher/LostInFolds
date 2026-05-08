# Datasets

We are using multi-rater medical image segmentation datasets for our investigations.
Each dataset should be prepared in the following manner:
- imagesTr: images used for training
- labelsTr: labels used for training, multi-rater masks (one label per rater)
- imagesTs: images used for testing
- labelsTs: labels used for testing, multi-rater masks (one label per rater)
- imagesOodTs: images used for OoD-testing
- labelsOoDTs: labels used for OoD-testing, multi-rater masks (one label per rater)

Set the nnU-Net paths to local directories before preprocessing:

```bash
export nnUNet_raw="<NNUNET_RAW>"
export nnUNet_preprocessed="<NNUNET_PREPROCESSED>"
export nnUNet_results="<NNUNET_RESULTS>"
```

Raw datasets, converted nnU-Net folders, preprocessed data, predictions, and model checkpoints are local artifacts and should not be committed.

## GleasonXAI

Dataset downloaded from [here](https://springernature.figshare.com/articles/dataset/Pathologist-like_explainable_AI_for_interpretable_Gleason_grading_in_prostate_cancer/27301845) (TissueArray images and all refined multi-rater annotations), [here](https://gleason2019.grand-challenge.org/Register/) (Gleason2019 images) and [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP) (Harvard Dataverse images).

Multi-rater labels are generated for all raters and stored under the per-subdataset all_raters folders.

Subdatasets are:
- TissueArray: 396 samples
- HarvardDataverse: 476 samples
- Gleason2019: 143 samples

Test set composition:
Subdatasets TissueArray (396) and HarvardDataverse (476) compose the main dataset.
We perform a 80:20 splitting to obtain and train/val and a test set.

OoD test set:
Subdataset Gleason2019 (143) is completely kept as a OoD dataset for testing.

### Preparation of GleasonXAI dataset

Prepare GleasonXAI to nnUNet_raw data format:
- Adjust paths/arguments in [src/data/gleasonxai/prepare.py](gleasonxai/prepare.py).
- Generate labels for all raters:
```
python3 ./src/data/gleasonxai/prepare.py --task generate_labels --raw_data_dir <RAW_DIR> --generate_labels_output_dir <LABELS_OUT>
```
- Convert images:
```
python3 ./src/data/gleasonxai/prepare.py --task convert_images --raw_data_dir <RAW_DIR> --generate_labels_output_dir <LABELS_OUT> --convert_images_output_dir <IMAGES_OUT> --convert_images_mode jpg_to_png
```
- Create nnUNet_raw with all raters (dataset.json contains dataset entries for training samples only):
```
python3 ./src/data/gleasonxai/prepare.py --task to_nnunet_raw_dataset --raw_data_dir <RAW_DIR> --nnunet_dataset_id <ID> --nnunet_labels_input_dir <LABELS_OUT> --nnunet_images_input_dir <IMAGES_OUT> --nnunet_output_dir <NNUNET_RAW_OUT> --main_ds tissue_array harvard_dataverse --ood_ds gleason2019
```

Preprocessed GleasonXAI data:
- set environment variables:
```
export nnUNet_raw="<NNUNET_RAW>"
export nnUNet_preprocessed="<NNUNET_PREPROCESSED>"
export nnUNet_results="<NNUNET_RESULTS>"
```
- start preprocessing: `nnUNetv2_plan_and_preprocess -d 003 -c 2d -pl nnUNetPlannerResEncM --verify_dataset_integrity -np 10`

## RIGA

Dataset downloaded from [here](https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z).

Place the downloaded RIGA files under `<RAW_DATA_DIR>/img_segmask_tif/`, preserving the original subdataset folders.
Data are separated according to their originating subdatasets.

Subdatasets are:
- BinRushed: 195 samples
- Magrabia: 94 samples
- MESSIDOR: 460 samples

Test set composition:
Subdatasets BinRushed (195) and MESSIDOR (460) compose the main dataset.
We perform a 80:20 splitting to obtain and train/val and a test set.

OoD test set:
Subdataset Magrabia (94) is completely kept as a OoD dataset for testing.

### Preparation of RIGA dataset

Prepare RIGA to nnUNet_raw data format:
- Adjust paths in [src/data/riga/prepare.py](riga/prepare.py).
- Execute `python3 ./src/data/riga/prepare.py`.

Preprocessed RIGA data:
- set environment variables:
```
export nnUNet_raw="<NNUNET_RAW>"
export nnUNet_preprocessed="<NNUNET_PREPROCESSED>"
export nnUNet_results="<NNUNET_RESULTS>"
```
- start preprocessing: `nnUNetv2_plan_and_preprocess -d 004 -c 2d -pl nnUNetPlannerResEncM --verify_dataset_integrity -np 10`

## CURVAS

Dataset downloaded from [here](https://zenodo.org/records/13767408).

### Preparation of CURVAS dataset

Prepare CURVAS to nnUNet_raw data format with all raters:
- Adjust paths/arguments in [src/data/curvas/prepare.py](curvas/prepare.py).
- Ensure CURVAS24_groups.xlsx is located in the raw data root (CURVAS2024/CURVAS24_groups.xlsx) and separates patients into groups a, b, c.
- Run the preparer (groups a+b are split 80:20 into train/test, group c is OOD; dataset.json contains dataset entries for training samples only):
```
python3 ./src/data/curvas/prepare.py --task to_nnunet_raw_dataset --raw_data_dir <CURVAS2024_DIR> --nnunet_dataset_id <ID> --nnunet_output_dir <NNUNET_RAW_OUT>
```

Preprocessed CURVAS data:
- set environment variables:
```
export nnUNet_raw="<NNUNET_RAW>"
export nnUNet_preprocessed="<NNUNET_PREPROCESSED>"
export nnUNet_results="<NNUNET_RESULTS>"
```
- start preprocessing: `nnUNetv2_plan_and_preprocess -d 005 -c 2d -pl nnUNetPlannerResEncM --verify_dataset_integrity -np 10`

## Gold Atlas

...
