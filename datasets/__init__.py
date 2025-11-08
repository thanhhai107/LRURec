from .ml_1m import ML1MDataset
from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .video import VideoDataset
from .sports import SportsDataset
from .steam import SteamDataset
from .xlong import XLongDataset


DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    VideoDataset.code(): VideoDataset,
    SportsDataset.code(): SportsDataset,
    SteamDataset.code(): SteamDataset,
    XLongDataset.code(): XLongDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
