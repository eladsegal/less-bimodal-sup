from datasets import load
import inspect
import logging

logger = logging.getLogger(__name__)

original_load_dataset_builder = load.load_dataset_builder


def patched_load_dataset_builder(*args, **kwargs):
    builder_instance = original_load_dataset_builder(*args, **kwargs)
    logger.info(inspect.getfile(builder_instance.__class__))
    return builder_instance


load.load_dataset_builder = patched_load_dataset_builder
