from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import gzip
import json
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class BeautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'beauty'

    @classmethod
    def url(cls):
        return 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['beauty_reviews.json.gz']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpfile = tmproot.joinpath('file')
        download(self.url(), tmpfile)
        os.makedirs(folder_path, exist_ok=True)
        shutil.move(tmpfile, folder_path.joinpath(self.all_raw_file_names()[0]))
        shutil.rmtree(tmproot)
        print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.remove_immediate_repeats(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        
        data = []
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                try:
                    review = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    review = ast.literal_eval(line.decode('utf-8'))
                data.append({
                    'uid': review['reviewerID'],
                    'sid': review['asin'],
                    'rating': review['overall'],
                    'timestamp': review['unixReviewTime']
                })
        
        df = pd.DataFrame(data)
        return df
