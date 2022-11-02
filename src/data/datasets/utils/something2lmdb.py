# Based on https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
# Need to check out https://www.pankesh.com/_posts/2019-05-18-efficiently-storing-and-retrieving-image-datasets.html

from typing import Collection, Optional
from abc import abstractmethod
import collections

from functools import cache
import os
from pathlib import Path
import shutil
import torch.utils.data as data
import lmdb
import pickle
from concurrent.futures import ThreadPoolExecutor

from datasets.fingerprint import Hasher

from src.pl.callbacks.progress import tqdm
from utils.redis_lock import redis_lock

import logging

logger = logging.getLogger(__name__)


class Something2Lmdb(data.Dataset):
    def __init__(
        self,
        input_path: str,
        requested_keys: Optional[Collection] = None,
        from_scratch: bool = False,
        use_cache_for_get: bool = False,  # Don't use cache in multiprocessing. Instead, just load everything in the beginning without lmdb
        all_in_memory: bool = False,
    ):
        input_path = self.input_path = os.path.realpath(input_path)

        if requested_keys is None:
            fingerprint = None
        else:
            hasher = Hasher()
            for key in sorted(requested_keys):
                hasher.update(key)
            fingerprint = hasher.hexdigest()

        lmdb_path = self.lmdb_path = _get_lmdb_path(input_path, fingerprint)

        if os.path.exists(lmdb_path) and from_scratch:
            for path in [lmdb_path, lmdb_path + "-lock"]:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

        with redis_lock(lmdb_path):
            if not os.path.exists(lmdb_path):
                logger.info(f"No LMDB, needs to create a new one")
                self.input2lmdb(
                    input_path=input_path,
                    output_path=lmdb_path,
                    requested_keys=requested_keys,
                )
            else:
                logger.info(f"Using existing lmdb from {lmdb_path}")

        self.env = None
        self.txn = None
        self.start_reading()
        self.length = loads(self.txn.get(b"__len__"))
        self.keys = loads(self.txn.get(b"__keys__"))
        self.end_reading()

        if requested_keys is not None:
            missing_keys = sorted(list(set(requested_keys) - set(self.keys)))
            if len(missing_keys) > 0:
                raise Exception(f"The following features are missing in {input_path}: {missing_keys}")

        self._all_in_memory = all_in_memory
        if self._all_in_memory:
            self._use_cache_for_get = False
            self.start_reading()
            self._lmdb_values = {}
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = tqdm(
                    executor.map(lambda k: (k, loads(self.txn.get(k.encode()))), self.keys),
                    desc="Loading LMDB values to memory (Multi-threaded)",
                    total=len(self.keys),
                )
                for id_, value in results:
                    self._lmdb_values[id_] = value
            self.end_reading()
        else:
            self._use_cache_for_get = use_cache_for_get

        assert self.env is None and self.txn is None  # Since we are going to serialize and deserialize after init

    @cache
    def __actual__getitem__(self, id_):
        if self._all_in_memory:
            lmdb_value = self._lmdb_values[id_]
        else:
            self.start_reading()
            byteflow = self.txn.get(id_.encode())
            lmdb_value = loads(byteflow)

        processed_value = self._process_lmdb_value(lmdb_value)

        return processed_value

    def __getitem__(self, id_):
        if self._use_cache_for_get:
            return self.__actual__getitem__(id_)
        else:
            return self.__actual__getitem__.__wrapped__(self, id_)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.lmdb_path + ")"

    def __del__(self):
        self.end_reading()

    def start_reading(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)

    def end_reading(self):
        if self.txn is not None:
            self.txn.commit()
            self.txn = None
        if self.env is not None:
            self.env.close()
            self.env = None

    @abstractmethod
    def _process_lmdb_value(self, lmdb_value):
        ...

    @abstractmethod
    def _get_input_elements_iterator(self, input_path, requested_keys: Optional[Collection]):
        ...

    def input2lmdb(
        self,
        input_path,
        output_path,
        requested_keys: Optional[Collection],
        write_frequency=5000,
    ):
        lmdb_path = output_path

        logger.info(f"Generate LMDB from {input_path} to {lmdb_path}")
        db_size = 1099511627776  # Arbitrary
        db = lmdb.open(lmdb_path, subdir=True, map_size=db_size, readonly=False, meminit=False, map_async=True)

        input_elements_iterator = self._get_input_elements_iterator(input_path, requested_keys)

        txn = db.begin(write=True)
        keys = []
        for idx, (key, element) in enumerate(tqdm(input_elements_iterator)):
            if key is None:
                continue
            keys.append(key)
            txn.put(key.encode(), dumps(element))
            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)

        # finish iterating through dataset
        txn.commit()
        with db.begin(write=True) as txn:
            txn.put(b"__keys__", dumps(keys))
            txn.put(b"__len__", dumps(len(keys)))

        logger.info("Flushing database ...")
        db.sync()
        db.close()


def loads(obj):
    return pickle.loads(obj)


def dumps(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


class IterableWithLength(collections.abc.Iterable):
    def __init__(self, iterator, length):
        self._iterator = iterator
        self._length = length

    def __len__(self):
        return self._length

    def __iter__(self):
        return self._iterator


def _get_lmdb_path(input_path, fingerprint: Optional[str]):
    fingerprint_suffix = f"_{fingerprint}" if fingerprint is not None else ""
    return os.path.abspath(
        os.path.join(
            os.path.dirname(input_path),
            f"{Path(input_path).stem}{Path(input_path).suffix.replace('.', '_')}{fingerprint_suffix}.lmdb",
        )
    )
