from tqdm import tqdm
import os
import threading
from typing import Callable, List, Tuple, Optional
import glob
from pathlib import Path
import csv
import requests
import math
import io

from PIL import Image

import fire

from google.cloud import storage
from tqdm import tqdm


MAX_FILES_PER_DIRECTORY = 10000


def get_dir(split: str, index: int):
    start = MAX_FILES_PER_DIRECTORY * (index // MAX_FILES_PER_DIRECTORY)
    end = start + MAX_FILES_PER_DIRECTORY
    return os.path.join(split, f"{start:08d}_{end:08d}")


def save_indices(split: str, thread_index: int, indices: List[int], name: str):
    file_name = f"{split}_{name}_indices{f'_{thread_index}' if thread_index != -1 else ''}.txt"
    with open(file_name, "w") as f:
        f.write("\n".join(map(str, indices)) + ("\n" if len(indices) > 0 else ""))


def merge_indices_files(split: str, num_threads: int, name: str):
    all_indices = []
    for i in range(num_threads):
        file_name = f"{split}_{name}_indices_{i}.txt"
        if os.path.isfile(file_name):
            with open(file_name) as f:
                all_indices.extend(int(x.strip()) for x in f.readlines())
            os.remove(file_name)
    save_indices(split, -1, all_indices, name)


####################
# GCLOUD
####################
BUCKET_NAME = "conceptual_captions"


def gcloud_upload_file(file_name, bucket):
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)


def gcloud_download_file(file_name, bucket):
    blob = bucket.blob(file_name)
    blob.download_to_filename(file_name)


def get_name_by_action(action: Callable):
    return f"{'upload' if action == gcloud_upload_file else 'download'}_failed"


def gcloud_thread(
    main_action: Callable,
    thread_index: int,
    image_indices: List[Tuple[List[str], int]],
    progress: tqdm,
    lock: Optional[threading.Lock],
    split: str,
):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    failed_indices = []
    for i, index in enumerate(image_indices):
        out_dir = get_dir(split, index)
        name = f"{index:08d}"
        out_path = os.path.join(out_dir, f"{name}.jpg")
        try:
            main_action(out_path, bucket)
        except BaseException:
            failed_indices.append(index)

        if i % 1000 == 0 and i > 0:
            save_indices(split, thread_index, failed_indices, get_name_by_action(main_action))

        interval = 10
        if i % interval == 0 and i > 0:
            if lock is not None:
                lock.acquire()
                try:
                    progress.update(interval)
                finally:
                    lock.release()
            else:
                progress.update(interval)

    save_indices(split, thread_index, failed_indices, get_name_by_action(main_action))


def gcloud_run(num_threads: int, path: str, main_action: Callable):
    os.chdir(path)
    for split in ("validation", "train"):
        indices_file = f"{split}_valid_image_indices.txt"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        main_action(indices_file, bucket)

        with open(indices_file, "r") as f:
            image_indices = [int(line.strip()) for line in f.readlines()]

        if main_action == gcloud_upload_file:
            uploaded_indices = set(
                int(Path(blob.name).stem)
                for blob in tqdm(
                    storage_client.list_blobs(BUCKET_NAME, prefix=f"{split}/"), desc="Getting uploaded indices"
                )
            )
            image_indices = [index for index in (set(image_indices) - uploaded_indices)]
        elif main_action == gcloud_download_file:
            downloaded_indices = set(
                int(Path(x).stem) for x in glob.iglob(os.path.join(split, "**/*.jpg"), recursive=True)
            )
            image_indices = [index for index in (set(image_indices) - downloaded_indices)]

        if len(image_indices) != 0:
            for i in range(math.ceil((max(image_indices) + 1) / MAX_FILES_PER_DIRECTORY)):
                os.makedirs(get_dir(split, i * MAX_FILES_PER_DIRECTORY), exist_ok=True)

            progress = tqdm(total=len(image_indices))
            if num_threads == 1:
                gcloud_thread(main_action, 0, image_indices, progress, None, split)
            else:
                groups = []
                threads = []
                lock = threading.Lock()
                split_size = max(1, len(image_indices) // num_threads)
                for i in range(num_threads):
                    groups.append(image_indices[i * split_size : (i + 1) * split_size])
                for i in range(num_threads):
                    threads.append(
                        threading.Thread(target=gcloud_thread, args=(main_action, i, groups[i], progress, lock, split))
                    )
                for i in range(num_threads):
                    threads[i].start()
                for i in range(num_threads):
                    threads[i].join()
            progress.close()

        merge_indices_files(split, num_threads, get_name_by_action(main_action))


####################
# URL DOWNLOAD
####################
def download_image(url: str, out_path: str, timeout=10):
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 200:
                image_bytes = bytearray()
                for chunk in r:
                    image_bytes.extend(chunk)

                Image.open(io.BytesIO(image_bytes))

                with open(out_path, "wb") as f:
                    f.write(image_bytes)

                return True
            return False
    except BaseException:
        return False


def url_download_thread(
    urls: List[Tuple[List[str], int]],
    thread_index: int,
    progress: tqdm,
    lock: Optional[threading.Lock],
    split: str,
):
    valid_image_indices = []
    for i in range(0, len(urls)):
        (caption, url), index = urls[i]
        out_dir = get_dir(split, index)
        name = f"{index:08d}"
        out_path = os.path.join(out_dir, f"{name}.jpg")

        if download_image(url, out_path):
            valid_image_indices.append(index)

        if i % 1000 == 0 and i > 0:
            save_indices(split, thread_index, valid_image_indices, "valid_image")

        interval = 10
        if i % interval == 0 and i > 0:
            if lock is not None:
                lock.acquire()
                try:
                    progress.update(interval)
                finally:
                    lock.release()
            else:
                progress.update(interval)

    save_indices(split, thread_index, valid_image_indices, "valid_image")


def url_download_run(num_threads: int, path: str):
    os.chdir(path)
    for split in ("validation", "train"):
        urls = []
        os.makedirs(split, exist_ok=True)
        if split == "train":
            tsv_path = f"Train_GCC-training.tsv"
        else:
            tsv_path = f"Validation_GCC-1.1.0-Validation.tsv"
        with open(tsv_path) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for i, row in enumerate(read_tsv):
                urls.append((row, i))

        for i in range(math.ceil(len(urls) / MAX_FILES_PER_DIRECTORY)):
            os.makedirs(get_dir(split, i * MAX_FILES_PER_DIRECTORY), exist_ok=True)

        downloaded_indices = set(
            int(Path(x).stem) for x in glob.iglob(os.path.join(split, "**/*.jpg"), recursive=True)
        )
        urls = [t for t in urls if t[1] not in downloaded_indices]

        progress = tqdm(total=len(urls))
        if num_threads == 1:
            url_download_thread(urls, 0, progress, None, split)
        else:
            groups = []
            threads = []
            lock = threading.Lock()
            split_size = max(1, len(urls) // num_threads)
            for i in range(num_threads):
                groups.append(urls[i * split_size : (i + 1) * split_size])
            for i in range(num_threads):
                threads.append(
                    threading.Thread(target=url_download_thread, args=(groups[i], i, progress, lock, split))
                )
            for i in range(num_threads):
                threads[i].start()
            for i in range(num_threads):
                threads[i].join()
        progress.close()

        merge_indices_files(split, num_threads, "valid_image")


class UploadDownloadFromGCloud:
    def __init__(self, path: str, num_threads: int = 16):
        self.path = path
        self.num_threads = num_threads
        if num_threads > 48:
            raise ValueError("num_threads > 48 might cause errors (open files limit and API errors)")

    def gcloud_upload(self):
        gcloud_run(self.num_threads, self.path, gcloud_upload_file)

    def gcloud_download(self):
        gcloud_run(self.num_threads, self.path, gcloud_download_file)

    def url_download(self):
        url_download_run(self.num_threads, self.path)


if __name__ == "__main__":
    fire.Fire(UploadDownloadFromGCloud)
