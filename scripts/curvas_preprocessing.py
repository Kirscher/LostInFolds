import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import SimpleITK as sitk

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate STAPLE consensus (seg + prob) for CURVASv1 dataset.\n\n"
            "Example:\n"
            "  python curvas_preprocessing.py --input_dir /path/to/CURVASv1 "
            "--threshold 0.5 --min_annotations 3 --num_workers 8\n"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory of CURVASv1 (contains training_set, validation_set, testing_set)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to STAPLE probability map"
    )
    parser.add_argument(
        "--min_annotations",
        type=int,
        default=3,
        help="Minimum number of annotations required"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing consensus files"
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def collect_annotation_files(patient_dir: Path):
    return sorted(patient_dir.glob("annotation_*.nii.gz"))


def run_staple(annotation_paths):
    images = [sitk.ReadImage(str(p), sitk.sitkUInt8) for p in annotation_paths]
    staple = sitk.STAPLEImageFilter()
    prob_map = staple.Execute(images)
    return prob_map


def _spacing_close(sp1, sp2, tol=1e-4):
    """
    Return True if two spacing tuples are equal within a tolerance.

    This avoids skipping annotations due to tiny floating-point differences.
    """
    if sp1 is None or sp2 is None:
        return False
    if len(sp1) != len(sp2):
        return False
    return all(abs(a - b) <= tol for a, b in zip(sp1, sp2))


def process_patient(patient_dir: Path, args):
    seg_out = patient_dir / "consensus_seg_STAPLE.nii.gz"
    prob_out = patient_dir / "consensus_prob_STAPLE.nii.gz"

    if not args.overwrite and seg_out.exists() and prob_out.exists():
        logging.info(f"{patient_dir.name}: consensus exists, skipping")
        return None

    image_path = patient_dir / "image.nii.gz"
    if not image_path.exists():
        logging.warning(f"{patient_dir.name}: missing image.nii.gz, skipping")
        return None

    annotation_paths = collect_annotation_files(patient_dir)
    if len(annotation_paths) < args.min_annotations:
        logging.warning(
            f"{patient_dir.name}: only {len(annotation_paths)} annotations, skipping"
        )
        return None

    try:
        # Reference image for spatial checks and metadata
        ref_img = sitk.ReadImage(str(image_path))
        ref_size = ref_img.GetSize()
        ref_spacing = ref_img.GetSpacing()

        valid_annotation_paths = []
        for p in annotation_paths:
            try:
                ann = sitk.ReadImage(str(p), sitk.sitkUInt8)
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
                valid_annotation_paths.append(p)
            except Exception as e:
                logging.warning(
                    f"{patient_dir.name}: failed to read annotation {p.name} ({e}), skipping"
                )

        if len(valid_annotation_paths) < args.min_annotations:
            logging.warning(
                f"{patient_dir.name}: only {len(valid_annotation_paths)} valid annotations after QC, skipping"
            )
            return None

        # Run STAPLE
        prob_map = run_staple(valid_annotation_paths)

        # Threshold to binary segmentation
        seg = sitk.BinaryThreshold(
            prob_map,
            lowerThreshold=args.threshold,
            upperThreshold=1.0,
            insideValue=1,
            outsideValue=0,
        )

        # Copy spatial metadata from reference image
        prob_map.CopyInformation(ref_img)
        seg.CopyInformation(ref_img)

        # Write outputs
        sitk.WriteImage(prob_map, str(prob_out))
        sitk.WriteImage(seg, str(seg_out))

        logging.info(
            f"{patient_dir.name}: wrote STAPLE prob + seg ({len(valid_annotation_paths)} annotations)"
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

    splits = ["training_set", "validation_set", "testing_set"]
    patient_dirs = []

    for split in splits:
        split_dir = args.input_dir / split
        if not split_dir.exists():
            logging.warning(f"Missing split folder: {split}")
            continue
        patient_dirs.extend([p for p in split_dir.iterdir() if p.is_dir()])

    logging.info(f"Found {len(patient_dirs)} patient folders")

    processed = []

    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_patient, p, args): p
                for p in patient_dirs
            }
            for f in tqdm(as_completed(futures), total=len(futures)):
                res = f.result()
                if res:
                    processed.append(res)
    else:
        for p in tqdm(patient_dirs, desc="Processing patients"):
            res = process_patient(p, args)
            if res:
                processed.append(res)

    logging.info(f"Finished. Generated consensus for {len(processed)} patients.")


if __name__ == "__main__":
    main()
