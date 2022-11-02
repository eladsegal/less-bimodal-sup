from typing import Dict, Any, Optional, Union
from collections import defaultdict
from collections.abc import Mapping

from src.data.dataset_containers import (
    BaseDatasetContainer,
    ComplexDatasetContainer,
    MultiDatasetContainer,
    HfDatasetContainer,
)
from utils.general import resolve_relative_paths
from src.utils.fallbacks import handle_fallback_per_split

from src.data.datasets.image import ImageFolder, ImageFolderToLmdb, ImageFile


class ImagesDatasetContainer(BaseDatasetContainer):
    def __init__(
        self,
        filtering_dataset_container: Optional[Union[str, HfDatasetContainer]] = None,
        images_base_dir: Optional[Dict[str, Any]] = None,
        image_transform=None,  # TODO: Currently does not allow to get raw_datasets
        lmdb: Optional[Union[bool, Dict[str, Any]]] = None,
        image_folder_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._filtering_dataset_container = filtering_dataset_container
        self._images_base_dir = resolve_relative_paths(images_base_dir)

        self._image_transform = handle_fallback_per_split(image_transform)

        self._lmdb = None
        if lmdb:
            self._lmdb = {
                "from_scratch": False,
                "use_cache_for_get": False,
            }
            if isinstance(lmdb, Mapping):
                self._lmdb.update(lmdb)

        self._image_folder_kwargs = image_folder_kwargs if image_folder_kwargs is not None else {}

    def _setup(
        self,
        parent_container: Optional[ComplexDatasetContainer] = None,
        additional_kwargs: Dict[str, Any] = None,
    ):
        if self._filtering_dataset_container is not None and isinstance(self._filtering_dataset_container, str):
            self._filtering_dataset_container = parent_container[self._filtering_dataset_container]
        if "image_transform" in additional_kwargs:
            image_transform = handle_fallback_per_split(additional_kwargs["image_transform"])
        else:
            image_transform = self._image_transform

        if self._images_base_dir is not None:
            self.ready_datasets = {}
            for split, images_base_dir in self._images_base_dir.items():
                current_image_transform = image_transform[split]

                keys = None
                if self._filtering_dataset_container is not None:
                    filtering_dataset = self._filtering_dataset_container.ready_datasets[split]

                    image_columns = [
                        column for column in filtering_dataset.column_names if column.startswith("image_file_name")
                    ]
                    keys = dict.fromkeys(
                        [
                            image_file_name
                            for image_column in image_columns
                            for image_file_name in filtering_dataset[image_column]
                        ]
                    )
                    if len(keys) == 0:
                        raise ValueError(
                            f"No image file names found in the {split} split for the {self._filtering_dataset_container.dataset_name} dataset"
                        )

                if self._lmdb is not None:
                    self.ready_datasets[split] = ImageFolderToLmdb(
                        input_path=images_base_dir,
                        requested_keys=keys,
                        **self._lmdb,
                        transform=current_image_transform,
                    )
                else:
                    self.ready_datasets[split] = ImageFolder(
                        images_base_dir,
                        transform=current_image_transform,
                        requested_keys=keys,
                        **self._image_folder_kwargs,
                    )
        else:
            if len(image_transform) == 0:
                self.ready_datasets = defaultdict(lambda: ImageFile(transform=image_transform.default_factory()))
            else:
                self.ready_datasets = {}
                for split, current_image_transform in image_transform.items():
                    self.ready_datasets[split] = ImageFile(transform=current_image_transform)

        self.raw_datasets = dict(self.ready_datasets)
