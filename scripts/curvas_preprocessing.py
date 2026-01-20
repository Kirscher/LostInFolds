import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import List, Sequence, Optional

import numpy as np
import SimpleITK as sitk

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


@dataclass
class ProcessingConfig:
    """
    Lightweight, picklable configuration passed into worker processes.
    """

    threshold: float
    min_annotations: int
    overwrite: bool
    labels: Optional[Sequence[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate STAPLE consensus (seg + prob) for CURVAS dataset.\n\n"
            "Supports multi-class masks by running STAPLE per label and fusing "
            "probabilities via argmax.\n\n"
            "Example:\n"
            "  python curvas_preprocessing.py --input_dir /path/to/CURVAS "
            "--threshold 0.5 --min_annotations 3 --num_workers 8\n"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory of CURVAS (contains training_set, validation_set, testing_set)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold applied per-class; voxels with max class "
            "probability below this are set to background."
        ),
    )
    parser.add_argument(
        "--min_annotations",
        type=int,
        default=3,
        help="Minimum number of annotations required after QC",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing consensus files",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of foreground labels to fuse (e.g. 1 2 3). "
            "If omitted, labels are auto-detected from the annotations (excluding 0)."
        ),
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def collect_annotation_files(patient_dir: Path) -> List[Path]:
    return sorted(patient_dir.glob("annotation_*.nii.gz"))


def _spacing_close(sp1: Sequence[float], sp2: Sequence[float], tol: float = 1e-4) -> bool:
    """
    Return True if two spacing tuples are equal within a tolerance.

    This avoids skipping annotations due to tiny floating-point differences.
    """
    if sp1 is None or sp2 is None:
        return False
    if len(sp1) != len(sp2):
        return False
    return all(abs(a - b) <= tol for a, b in zip(sp1, sp2))


def _detect_labels(annotation: sitk.Image) -> List[int]:
    """
    Detect non-zero labels present in a single annotation image.

    Uses SimpleITK LabelStatisticsImageFilter to avoid materializing full
    3D volumes into NumPy and calling np.unique on them. The label image
    is explicitly cast to an integer type so float-encoded labels work.
    """
    stats = sitk.LabelStatisticsImageFilter()
    # Cast to an integer type for the label image; intensity values are irrelevant here.
    label_img = sitk.Cast(annotation, sitk.sitkUInt16)
    stats.Execute(label_img, label_img)

    labels: List[int] = []
    for lab in stats.GetLabels():
        iv = int(lab)
        if iv != 0:
            labels.append(iv)

    return sorted(set(labels))


def _run_staple_for_label(annotations: Sequence[sitk.Image], label: int) -> sitk.Image:
    """
    Run STAPLE on binary masks for a single label.
    """
    images: List[sitk.Image] = []
    for ann in annotations:
        mask = sitk.Equal(ann, label)  # 1 where ann == label, 0 elsewhere
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        images.append(mask)

    staple = sitk.STAPLEImageFilter()
    prob_map = staple.Execute(images)
    return prob_map


def write_nifti(image: sitk.Image, path: Path, compressed: bool = True) -> None:
    """
    Helper to explicitly control NIfTI compression when writing outputs.
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    if compressed:
        writer.UseCompressionOn()
    else:
        writer.UseCompressionOff()
    writer.Execute(image)


def process_patient(patient_dir: Path, config: ProcessingConfig):
    seg_out = patient_dir / "consensus_seg_STAPLE.nii.gz"
    prob_out = patient_dir / "consensus_prob_STAPLE.nii.gz"

    if not config.overwrite and seg_out.exists() and prob_out.exists():
        logging.info(f"{patient_dir.name}: consensus exists, skipping")
        return None

    image_path = patient_dir / "image.nii.gz"
    if not image_path.exists():
        logging.warning(f"{patient_dir.name}: missing image.nii.gz, skipping")
        return None

    annotation_paths = collect_annotation_files(patient_dir)
    if len(annotation_paths) < config.min_annotations:
        logging.warning(
            f"{patient_dir.name}: only {len(annotation_paths)} annotations, skipping"
        )
        return None

    try:
        # Reference image for spatial checks and metadata
        ref_img = sitk.ReadImage(str(image_path))
        ref_size = ref_img.GetSize()
        ref_spacing = ref_img.GetSpacing()

        valid_annotations_itk: List[sitk.Image] = []

        for p in annotation_paths:
            try:
                ann = sitk.ReadImage(str(p))
                ann_size = ann.GetSize()
                ann_spacing = ann.GetSpacing()
                same_size = ann_size == ref_size
                spacing_ok = _spacing_close(ann_spacing, ref_spacing)
                if (not same_size) or (not spacing_ok):
                    logging.warning(
                        f"{patient_dir.name}: annotation {p.name} has mismatched size/spacing, "
                        f"image size={ref_size}, ann size={ann_size}, "
                        f"image spacing={ref_spacing}, ann spacing={ann_spacing}; skipping"
                    )
                    continue

                valid_annotations_itk.append(ann)
            except Exception as e:
                logging.warning(
                    f"{patient_dir.name}: failed to read annotation {p.name} ({e}), skipping"
                )

        if len(valid_annotations_itk) < config.min_annotations:
            logging.warning(
                f"{patient_dir.name}: only {len(valid_annotations_itk)} valid annotations after QC, skipping"
            )
            return None

        logging.info(
            f"{patient_dir.name}: {len(valid_annotations_itk)}/{len(annotation_paths)} "
            f"annotations kept after QC"
        )

        # Determine labels to fuse
        if config.labels:
            labels = sorted(set(int(l) for l in config.labels if l != 0))
        else:
            # Detect labels from a single representative annotation image
            labels = _detect_labels(valid_annotations_itk[0])

        if not labels:
            logging.warning(f"{patient_dir.name}: no foreground labels found, skipping")
            return None

        # Run STAPLE per label and stream argmax over labels to avoid stacking
        max_prob: Optional[np.ndarray] = None
        seg_array: Optional[np.ndarray] = None

        for lab in labels:
            prob_map = _run_staple_for_label(valid_annotations_itk, lab)
            prob_map.CopyInformation(ref_img)

            p_arr = sitk.GetArrayFromImage(prob_map).astype(np.float32)

            if max_prob is None:
                max_prob = p_arr
                seg_array = np.full(p_arr.shape, lab, dtype=np.int16)
            else:
                update_mask = p_arr > max_prob
                max_prob[update_mask] = p_arr[update_mask]
                seg_array[update_mask] = lab

        if max_prob is None or seg_array is None:
            logging.warning(f"{patient_dir.name}: STAPLE produced no probabilities, skipping")
            return None

        # Apply threshold: background where max_prob < threshold
        below_thresh = max_prob < config.threshold
        seg_array[below_thresh] = 0

        # Convert back to images
        seg_img = sitk.GetImageFromArray(seg_array.astype(np.int16))
        seg_img.CopyInformation(ref_img)

        # Store probability of assigned label as a single scalar prob map
        prob_winner_img = sitk.GetImageFromArray(max_prob.astype(np.float32))
        prob_winner_img.CopyInformation(ref_img)

        # Write outputs with explicit compression control
        write_nifti(prob_winner_img, prob_out, compressed=True)
        write_nifti(seg_img, seg_out, compressed=True)

        logging.info(
            f"{patient_dir.name}: wrote multi-class STAPLE consensus "
            f"for labels {labels} using {len(valid_annotations_itk)} annotations"
        )
        return patient_dir.name

    except Exception as e:
        logging.error(f"{patient_dir.name}: failed ({e})")
        return None


def main():
    args = parse_args()
    setup_logging()

    # Prevent SimpleITK thread oversubscription
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    config = ProcessingConfig(
        threshold=args.threshold,
        min_annotations=args.min_annotations,
        overwrite=args.overwrite,
        labels=args.labels,
    )

    splits = ["training_set", "validation_set", "testing_set"]
    patient_dirs: List[Path] = []

    for split in splits:
        split_dir = args.input_dir / split
        if not split_dir.exists():
            logging.warning(f"Missing split folder: {split}")
            continue
        patient_dirs.extend([p for p in split_dir.iterdir() if p.is_dir()])

    logging.info(f"Found {len(patient_dirs)} patient folders")

    processed: List[str] = []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_patient, p, config): p
                for p in patient_dirs
            }
            for f in tqdm(as_completed(futures), total=len(futures)):
                res = f.result()
                if res:
                    processed.append(res)
    else:
        for p in tqdm(patient_dirs, desc="Processing patients"):
            res = process_patient(p, config)
            if res:
                processed.append(res)

    logging.info(f"Finished. Generated consensus for {len(processed)} patients.")


if __name__ == "__main__":
    main()
