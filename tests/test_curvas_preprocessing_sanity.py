from pathlib import Path

import SimpleITK as sitk

from src.data.curvas.prepare import process_patient, ProcessingConfig


def _write_dummy_image(path: Path, size=(4, 4, 4)):
    """Create and write a small 3D image for testing."""
    image = sitk.Image(size[0], size[1], size[2], sitk.sitkUInt8)
    image += 1  # set all voxels to 1
    sitk.WriteImage(image, str(path))


def test_process_patient_creates_consensus_files(tmp_path):
    """
    Quick sanity check that `process_patient` runs end-to-end on a toy case.

    This uses tiny synthetic images so that STAPLE is fast.
    """
    # Build a fake CURVAS-like directory structure
    input_dir = tmp_path
    split_dir = input_dir / "training_set"
    patient_dir = split_dir / "patient_001"
    patient_dir.mkdir(parents=True)

    # Reference image
    image_path = patient_dir / "image.nii.gz"
    _write_dummy_image(image_path)

    # Create several consistent annotation masks
    num_annotations = 3
    for i in range(num_annotations):
        ann_path = patient_dir / f"annotation_{i+1}.nii.gz"
        _write_dummy_image(ann_path)

    # Minimal config matching what `process_patient` expects
    config = ProcessingConfig(
        threshold=0.5,
        min_annotations=3,
        overwrite=True,
        labels=None,
    )

    # Run
    result = process_patient(patient_dir, config)

    seg_out = patient_dir / "consensus_seg_STAPLE.nii.gz"
    prob_out = patient_dir / "consensus_prob_STAPLE.nii.gz"

    # Sanity assertions
    assert result == patient_dir.name
    assert seg_out.exists()
    assert prob_out.exists()

    # Check that output geometry matches the reference image
    ref_img = sitk.ReadImage(str(image_path))
    seg_img = sitk.ReadImage(str(seg_out))
    prob_img = sitk.ReadImage(str(prob_out))

    assert seg_img.GetSize() == ref_img.GetSize()
    assert prob_img.GetSize() == ref_img.GetSize()
    assert seg_img.GetSpacing() == ref_img.GetSpacing()
    assert prob_img.GetSpacing() == ref_img.GetSpacing()

