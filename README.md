# LostInFolds

## Datasets
We are using multi-rater medical image segmentation datasets for our investigations.
Each dataset should be prepared in the following manner:
- imagesTr: images used for training
- labelsTr: labels used for training, should be consensus masks of multi-rater masks
- imagesTs: images used for testing
- labelsTs: labels used for testing
- imagesOodTs: images used for OoD-testing
- labelsOoDTs: labels used for OoD-testing

### GleasonXAI

Dataset downloaded from [here](https://springernature.figshare.com/articles/dataset/Pathologist-like_explainable_AI_for_interpretable_Gleason_grading_in_prostate_cancer/27301845) (TissueArray images and all refined milti-rater annotations), [here](https://gleason2019.grand-challenge.org/Register/) (Gleason2019 images) and [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP) (Harvard Dataverse images).

nnUNet_raw in ~/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/GleasonXAI/nnUNet_raw.
Data separated according (for FL) according to their originating subdatasets.

Multi-rater labels in /home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/GleasonXAI/generated_labels/*subdataset*/random_rater.

Subdatasets are:
- TissueArray: 396 samples
- HarvardDataverse: 476 samples
- Gleason2019: 143 samples

Test set composition:
Subdatasets TissueArray (396) and HarvardDataverse (476) compose the main dataset.
We perform a 80:20 splitting to obtain and train/val and a test set.

OoD test set:
Subdataset Gleason2019 (143) is completely kept as a OoD dataset for testing.

#### Preparation of GleasonXAI dataset

Prepare GleasonXAI to nnUNet_raw data format:
- Adjust paths in /src/data/gleasonxai/prepare.
- Execute `python3 ./src/data/gleasonxai/prepare`.

Preprocessed GleasonXAI data:
- set environmental variables:
```
export nnUNet_raw="/home/m391k/E132-Projekte/Projects/2026_Kirscher_LostInFolds/data/GleasonXAI"
export nnUNet_preprocessed="/home/m391k/cluster-data_all/t789r/preprocessed_data"
```
- start preprocessing: `nnUNetv2_plan_and_preprocess -d 001 -c 2d -pl nnUNetPlannerResEncM --verify_dataset_integrity -np 10`

### RIGA

Dataset downloaded from [here](https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z).

nnUNet_raw in ~/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/RIGA/nnUNet_raw.
Data separated according (for FL) according to their originating subdatasets.

Multi-rater labels in /home/m391k/E132-Projekte/Projects/2024_Bujotzek_Noisy-Seg-Label-Benchi/data/RIGA/img_segmask_tif/*subdataset*/.

Subdatasets are:
- BinRushed: 195 samples
- Magrabia: 94 samples
- MESSIDOR: 460 samples

Test set composition:
Subdatasets BinRushed (195) and MESSIDOR (460) compose the main dataset.
We perform a 80:20 splitting to obtain and train/val and a test set.

OoD test set:
Subdataset Magrabia (94) is completely kept as a OoD dataset for testing.


#### Preparation of RIGA dataset

Prepare RIGA to nnUNet_raw data format:
- Adjust paths in /src/data/riga/prepare.
- Execute `python3 ./src/data/riga/prepare`.

Preprocessed RIGA data:
- set environmental variables:
```
export nnUNet_raw="/home/m391k/E132-Projekte/Projects/2026_Kirscher_LostInFolds/data/RIGA"
export nnUNet_preprocessed="/home/m391k/cluster-data_all/t789r/preprocessed_data"
```
- start preprocessing: `nnUNetv2_plan_and_preprocess -d 002 -c 2d -pl nnUNetPlannerResEncM --verify_dataset_integrity -np 10`

### CURVAS
Dataset downloaded from [here](https://zenodo.org/records/13767408).

#### Preparation of CURVAS dataset

Generate STAPLE consensus masks for CURVAS:
- Adjust paths in /src/data/curvas/prepare.py.
- Execute `python3 ./src/data/curvas/prepare.py --input_dir /path/to/CURVAS --threshold 0.5 --min_annotations 3 --num_workers 8`.

### Gold Atlas
...

