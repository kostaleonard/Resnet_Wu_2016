"""ILSVRCDataset class."""

from dataset.dataset import Dataset


class ILSVRCDataset(Dataset):
    """Represents the ILSVRC2012 dataset."""

    def _load_or_download_dataset(self, verbose: bool = False) -> None:
        """Loads the dataset from a pre-existing path, or downloads it.
        :param verbose: whether to print info to stdout.
        """
        super()._load_or_download_dataset(verbose=verbose)
        # TODO check for dataset. If not found, download. Ask for user confirmation because imagenet is so huge.
        # TODO fill partition and labels.
