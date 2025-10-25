from segFM.DataLoaders import bagls, endoscapes, flare22, imed361, medical_decathlon, neopolyp, vfss, amos22
from segFM.DataLoaders import medsegbench, brain_em, toothfairy3, refuge2, brats_medical_decathlon
from segFM.BOB.src.BOB.train.dataset_converter import DatasetConverter
from segFM.logger import logger

def convert_3d_datasets(base_path):
    """
    Convert 3D datasets into a format suitable for object detection tasks.
    This function is a placeholder for future implementations of 3D dataset conversions.
    """
    datasets = [
        #toothfairy3.ToothFairy3(plot_prompts=False, tooth_only=True),
        #amos22.Amos22(split="train", plot_prompts=False, preprocess="random"),
        #flare22.Flare22(plot_prompts=False, preprocess="random"),
    ]

    for dataset in datasets:
        logger.info(f"Converting {dataset.__class__.__name__} dataset")

        # Create a DatasetConverter instance for the dataset
        converter = DatasetConverter(dataset, base_path=base_path, allow_empty=True)
        converter.convert(n_images=150)

def convert_2d_datasets(base_path):
    """
    Convert 2D datasets into a format suitable for object detection tasks.
    This function is a placeholder for future implementations of 2D dataset conversions.
    """
    datasets = [
        #refuge2.Refuge2Dataset(mode="box", plot_prompts=False, bbsize=0),
        #bagls.BAGLSImagesFull(mode="box"),
        neopolyp.NeopolypDataset(mode="box", plot_prompts=False, bbsize=0, split="train"),
        #brain_em.BrainMicroscopy(mode="box", plot_prompts=False, bbsize=0),
        #endoscapes.EndoscapesDataset(mode="box", plot_prompts=False, bbsize=0),
    ]

    for dataset in datasets:
        logger.info(f"Converting {dataset.__class__.__name__} dataset")

        # Create a DatasetConverter instance for the dataset
        converter = DatasetConverter(dataset, base_path=base_path, allow_empty=True)
        converter.convert(n_images=1000)


def convert_imed361(base_path):
    im = imed361.IMed361()

    for dataset_id in range(im.n_datasets):
        if dataset_id in [38, 40]:
            # SKIP DATASET THAT ARE NOT COMPLETELY LABELED
            # Example: MSD Spleen only has labels for spleen, not for other organs
            # HCC-TACE only has labels for liver (tumor) and aorta, not for other organs
            continue
        dataset = imed361.IMed361(dataset_id=dataset_id, mode="box", plot_prompts=True, bbsize=0)
        logger.info(f"Converting dataset {dataset_id} - {dataset.dataset_name}")

        # Create a DatasetConverter instance for the IMed361 dataset
        converter = DatasetConverter(dataset, base_path=base_path)
        converter.convert(n_images=1500)

def convert_medsegbench(base_path):
    """
    Convert the MedSegBench dataset into a format suitable for object detection tasks.
    """
    for dataset_id in range(33):
        if dataset_id == 16:  # Skip dataset 16 as it is not available
            continue
        # Create a MedSegBench dataset instance for the given dataset_id
        dataset = medsegbench.MedSegBench(dataset_id=dataset_id, mode="box", plot_prompts=False, bbsize=0)
        logger.info(f"Converting MedSegBench dataset {dataset_id} - {dataset.dataset_name}")

        # Create a DatasetConverter instance for the MedSegBench dataset
        converter = DatasetConverter(dataset, base_path=base_path)
        converter.convert(n_images=1200)

def convert_BRATS_Glioma(base_path):
    """
    Convert the BRATS Glioma dataset into a format suitable for object detection tasks.
    """
    dataset = brats_medical_decathlon.BratsMSDGlioma(yolo=False, transform=None, modality=0)
    logger.info(f"Converting BRATS Glioma dataset")

    # Create a DatasetConverter instance for the BRATS Glioma dataset
    converter = DatasetConverter(dataset, base_path=base_path, allow_empty=True)
    converter.convert(n_images=1000)

if __name__ == "__main__":
    base_path="BOB_dataset"  # Base path to save converted datasets

    logger.info("Starting dataset conversion...")

    # # Convert IMed361 dataset
    # convert_imed361(base_path)

    # # Convert MedSegBench dataset
    # convert_medsegbench(base_path)

    # # Convert 2D datasets - BAGLS, Neopolyp...
    # convert_2d_datasets(base_path)

    # Convert 3D datasets - Flare22, Amos22...
    # convert_3d_datasets(base_path)

    # convert_BRATS_Glioma(base_path)

    # logger.info("Dataset conversion completed.")
    # logger.info("Starting dataset combination...")
 
    converter = DatasetConverter(None, base_path=base_path)
    converter.combine_datasets(base_path)

    logger.info("Combined datasets into a single folder.")
    