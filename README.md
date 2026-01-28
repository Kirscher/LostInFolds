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


## Tasks

* $M$ ensemble outputs, $N$ reference segmentations
* $y_m$: Segmentation prediction $m$ of ensemble
* $y_n^*$: Reference segmentation $n$ of multiple reference segs
* $\bar{y}$: Mean segmentation prediction
* $\bar{y}^*$: Consensus reference seg
* $p_m$: Softmax prediction $m$ of ensemble
* $\bar{p}$: Mean softmax prediction

### Segmentation

**Segmentation Performance**  

$\text{Dice}(\bar{y}, \bar{y}^*)$

I.e. the Dice score between the consensus reference seg and the mean predicted seg


**Inter-model Agreement**  
...

### Uncertainty Task

| Task | Metric | Input Pred | Input GT | Other |
| ---- | ---- | ---- | ---- | ---- | 
Calibration | ACE | $\bar{y}$ | $y_0^*, ..., y_N^*$ | $\text{conf}=\max_{c} \bar{p}_c$ |
Calibration | BA-ECE | $\bar{y}$ | $y_0^*, ..., y_N^*$ | $\text{conf}=\max_{c} \bar{p}_c$ |
Calibration | SPACE | $\bar{y}$ | $y_0^*, ..., y_N^*$ | $\text{conf}=\max_{c} \bar{p}_c$ |
Ambiguity Modeling | NCC | $\mathbb{E}[H(p)]$ | $\mathbb{V}[y_i^*]$ |
Ambiguity Modeling | GED | $y_0, ..., y_M$ | $y_0^*, ..., y_N^*$ |
Failure Detection | AURC | xxx | xxx |$r = 1- \text{Dice}(\bar{y}, \bar{y}^*)$; $\text{conf} = \mathbb{E}[\{\text{Dice}(y_i, y_j)\}_{i \neq j}]$

-------------

**Calibration Task**

1. *Average Calibration Error (ACE)*
    
    Let $S_m$ denote the set of voxels whose confidence falls into bin $m$. The average confidence in bin $m$ is
    
    $\text{conf}_m = \frac{1}{|S_m|} \sum_{v \in S_m} \text{conf}(v)$,
    
    and the average accuracy in bin $m$, computed against $N$ reference segmentations $\{y_1^*, \ldots, y_N^*\}$, is
    
    $\text{acc}_m =
    \frac{1}{|S_m|}
    \sum_{v \in S_m}
    \left(
    \frac{1}{N}
    \sum_{n=1}^{N}
    \mathbb{I}(\hat y(v) = y_n^*(v))
    \right)$.
    
    The Average Calibration Error is then defined as
    
    $\text{ACE} = \frac{1}{M} \sum_{m=1}^{M} |\text{conf}_m - \text{acc}_m|$.

    
2. *Boundary-Aware Expected Calibration Error (BA-ECE)*

    For each voxel $x$, let $d(x)$ denote its shortest distance to the boundary. Distances are partitioned into $K$ bands $\{b_1, \ldots, b_K\}$, where
    
    $b_i = \{x \mid d(x) \in \Delta_i\}$,
    
    and $\Delta_i$ denotes the $i$-th distance interval.
    
    For each band $b_i$, the mean predicted uncertainty and mean observed error are computed as
    
    $\mu^{U}_{b_i} = \frac{1}{|b_i|} \sum_{x \in b_i} U(x), \qquad
    \mu^{E}_{b_i} = \frac{1}{|b_i|} \sum_{x \in b_i} E(x)$,
    
    where $U(x)$ is the predicted uncertainty and $E(x)$ is a binary error indicator.
    
    The Boundary-Aware Expected Calibration Error is then defined as
    
    $\text{BA-ECE} = \sum_{i=1}^{K} w_i \, \big| \mu^{U}_{b_i} - \mu^{E}_{b_i} \big|$,
    
    where $w_i$ is a distance-based weight inversely proportional to the average distance of voxels in band $b_i$ from the boundary. Larger penalties are thus assigned to miscalibration near object boundaries.


3. *Spatially-Aware Calibration Error (SPACE)*
    
    SPACE evaluates the local spatial agreement between a predicted uncertainty map $U$ and a binary error map $E$. Both maps are convolved with a Gaussian kernel $G_\sigma$ to obtain spatially smoothed representations.
    
    $\text{SPACE} =
    \text{mean} \left|
    (G_\sigma * U) - (G_\sigma * E)
    \right|$.
    
    Lower SPACE values indicate that predicted uncertainty better aligns with the spatial distribution of actual errors within local neighborhoods defined by the Gaussian kernel width $\sigma$.


-------------

**Ambiguity Modeling**

1. *NCC*

    The Normalized Cross Correlation is defined as 

    $\frac{1}{V\sigma_a\sigma_b}\sum_{v=1}^{V}(a(v) - \mu_a) \cdot (b(v) - \mu_b)$

    Here, $a$ is the reference uncertainty map, $b$ is the predicted uncertainty map, $V$ is the total number of pixels in the uncertainty maps, and $\mu$ and $\sigma$ are mean and standard deviation of the uncertainty maps. 

    The reference uncertainty map is calculated with the pixel variance of a pixel $y^*(v)$ for $N$ different segmentation raters $\{y^*_1(v),...,y^*_N(v)\}$:

    $\mathbb{V}_{p(D)}[y^*(v)] = \frac{1}{N}\sum_{n=1}^{N}(y^*_n(v) - \bar{y}(v))^2$

    The predicted uncertainty map is calculated via the expected entropy:

    $\mathbb{E}[H(p)] = \frac{1}{M} \sum_{m=1}^M (-\sum_{y\in Y} (p_m(y) \log p_m(y)))$  


2. *GED*

    The GED is defined as 

    $D_{\text{GED}}^2(p^*, p) = 2\mathbb{E}_{y^*\sim p^*, y\sim p}[d(y^*,y)] - \mathbb{E}_{y^*, y'^* \sim p^*}[d(y^*, y'^*)] - \mathbb{E}_{y, y' \sim p}[d(y, y')]$

    Here, $d(y^*, y'^*)$ is the distance between two reference segmentations, and $d(y, y')$ is the distance between two predicted segmentation variants. $p^*$ and $p$ are the respective reference and predicted distributions for the segmentations masks.

    As distance, the Dice score can be used as

    $d(x,y) = 1 - \text{Dice}(x,y)$

-------------

**Failure Detection**

For failure detection, we use the Area under the Risk-Coverage-Curve (AURC)

The risk is defined as 

$r = 1- \text{Dice}(\bar{y}, \bar{y}^*)$

The selective risk given a threshold $\tau$ and a confidence scoring function $\text{conf}$ is given as

$\text{Risk} = \frac{\sum_{i=1}^D r \cdot \mathbb{I}(\text{conf} \ge \tau)}{\sum_{i=1}^D\mathbb{I}(\text{conf} \ge \tau)}$

Where $D$ is the number of cases in the dataset.

The coverage is defined as the ratio of cases remaining after selection:

$\text{Coverage} = \frac{\sum_{i=1}^D\mathbb{I}(\text{conf} \ge \tau)}{D}$

As confidence function, we use the pairwise Dice score between the predicted segmentations:

$\text{conf} = \mathbb{E}[\{\text{Dice}(y_i, y_j)\}_{i \neq j}]$

The AURC based on a threshold list $\{\tau\}_{t=1}^T$ with $T$ values of a CSF that are sorted ascending can then be calculated as

$\text{AURC} = \sum_{t=1}^T(\text{Coverage}(\tau_t) - \text{Coverage}(\tau_{(t-1)})) \cdot (\text{Risk}(\tau_t) + \text{Risk}(\tau_{t-1}))/2$
