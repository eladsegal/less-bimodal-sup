from typing import Collection, Optional

from functools import cache
import os
import sys
import io
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import subprocess

import torch.utils.data as data
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from src.data.datasets.utils.something2lmdb import Something2Lmdb, IterableWithLength
from src.pl.callbacks.progress import tqdm
from utils.general import get_source_code_path, get_files_with_extensions

import logging

logger = logging.getLogger(__name__)


def read_image(path, is_clip=False):
    if urlparse(path).scheme != "":
        response = requests.get(path)
        img = _open_image(io.BytesIO(response.content), is_clip=is_clip)
    else:
        with open(path, "rb") as f:
            img = _open_image(f, is_clip=is_clip)
    return img


def _open_image(f, is_clip=False):
    return Image.open(f).convert("RGBA" if is_clip else "RGB")


class ImageFile(data.Dataset):
    def __init__(self, use_cache_for_get=False, transform=None, is_clip=False):
        self._use_cache_for_get = use_cache_for_get
        self._transform = transform
        self._is_clip = is_clip

    @cache
    def __actual__getitem__(self, file_path):
        img = read_image(file_path, is_clip=self._is_clip)
        if self._transform is not None:
            img = self._transform(img)
        return img

    # TODO: Prevent code duplication by creating a new class, MemoryCachedDataset?
    def __getitem__(self, file_path):
        if self._use_cache_for_get:
            func = self.__actual__getitem__
        else:
            func = self.__actual__getitem__.__wrapped__
        return func(self, file_path)


def double_fork_run(command):
    try:
        pid = os.fork()
        if pid > 0:
            return
    except OSError as e:
        raise Exception(f"fork #1 failed: {e.errno}, {e.strerror}")

    os.setsid()

    # do second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit()
    except OSError as e:
        raise Exception(f"fork #2 failed: {e.errno}, {e.strerror}")

    subprocess.Popen(command.split(), cwd=get_source_code_path())
    sys.exit()


class ImageFolder(data.Dataset):
    def __init__(
        self,
        image_folder_path: str,
        use_cache_for_get=False,  # If used in multiprocessing, load everything before the fork
        transform=None,
        is_clip=False,
        all_in_memory: bool = False,
        extension: str = ".jpg",
        requested_keys: Optional[Collection] = None,
    ):
        self.image_folder_path = image_folder_path
        self._is_clip = is_clip

        if requested_keys is not None:
            image_paths = [key_to_image_path(key, image_folder_path) for key in requested_keys]
        else:
            listing_path = os.path.join(image_folder_path, f"all_images_{extension}.txt")
            if os.path.exists(listing_path):
                with open(os.path.join(image_folder_path, "images.json"), "r") as f:
                    image_paths = f.readlines()
            else:
                logger.info(f"Getting all paths in {image_folder_path}. This may take a while.")
                image_paths = get_files_with_extensions(image_folder_path, [extension])
                with open(listing_path, "w") as f:
                    f.write("\n".join(image_paths))

        self._all_in_memory = all_in_memory
        if self._all_in_memory:
            self._use_cache_for_get = False
            self._logged_not_found = False

            # Local cache
            self._cache = {}
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = tqdm(
                    executor.map(
                        lambda image_path: (image_path, read_image(image_path, is_clip=self._is_clip)), image_paths
                    ),
                    desc="Loading images to memory (Multi-threaded)",
                    total=len(image_paths),
                )
                for image_path, img in results:
                    self._cache[image_path_to_key(image_path, image_folder_path)] = img
        else:
            self._use_cache_for_get = use_cache_for_get

        self._image_keys = [image_path_to_key(image_path, image_folder_path) for image_path in image_paths]

        self._transform = transform

        logger.info(f"Using image folder {image_folder_path}")

    @property
    def keys(self):
        return self._image_keys

    @cache
    def __actual__getitem__(self, file_name):
        if self._all_in_memory:
            # Local cache
            img = self._cache[file_name]
        else:
            img = read_image(os.path.join(self.image_folder_path, file_name), is_clip=self._is_clip)

        if self._transform is not None:
            img = self._transform(img)
        return img

    def __getitem__(self, file_name):
        if self._use_cache_for_get:
            return self.__actual__getitem__(file_name)
        else:
            return self.__actual__getitem__.__wrapped__(self, file_name)

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.image_folder_path + ")"


class ImageFolderToLmdb(Something2Lmdb):
    def __init__(self, transform=None, is_clip=False, extension: str = ".jpg", **kwargs):
        super().__init__(**kwargs)
        self._transform = transform
        self._is_clip = is_clip
        self._extension = extension

    def _process_lmdb_value(self, bytes):
        buf = io.BytesIO(bytes)
        buf.seek(0)
        img = _open_image(buf, is_clip=self._is_clip)
        if self._transform is not None:
            img = self._transform(img)
        return img

    def _get_input_elements_iterator(self, input_path, requested_keys: Optional[Collection]):
        if requested_keys is not None:
            image_paths = [key_to_image_path(key, input_path) for key in requested_keys]
        else:
            image_paths = get_files_with_extensions(input_path, [self._extension])

        def iterator():
            for image_path in image_paths:
                key = image_path_to_key(image_path, input_path)
                with open(image_path, "rb") as f:
                    image = f.read()
                yield key, image

        return IterableWithLength(iterator(), len(image_paths))


def image_path_to_key(image_path, input_path):
    trailing_slash_input_path = os.path.join(input_path, "")
    return image_path.replace(trailing_slash_input_path, "", 1)


def key_to_image_path(key, input_path):
    return os.path.join(input_path, key)
