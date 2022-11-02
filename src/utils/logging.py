import logging
from logging import Filter
import sys


def set_logger(level=logging.INFO, external_level=logging.INFO):
    logging.basicConfig(stream=sys.stdout, level=level, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    logging.getLogger("datasets.metric").setLevel(logging.WARNING)
    for logger_name in [
        "transformers",
        "pytorch_lightning",
        "datasets",
        "datasets.builder",
        "datasets.arrow_dataset",
        "wandb",
    ]:
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).propagate = True
        logging.getLogger(logger_name).setLevel(external_level)
    logging.getLogger("pottery").setLevel(logging.WARNING)


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.ERROR


def stdout_filter_errors():
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if handler.stream.name == "<stdout>":
            handler.addFilter(ErrorFilter())
            break


def write_uncaught_exceptions():
    root_logger = logging.getLogger()
    # write uncaught exceptions to the logs
    def excepthook(exctype, value, traceback):
        # For a KeyboardInterrupt, call the original exception handler.
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook


class StdoutLogger(object):
    tqdm_formatter = logging.Formatter("%(message)s")

    def __init__(self, logger, handlers):
        self.logger = logger
        self.handlers = handlers
        self.terminal = sys.stdout

    def __getattr__(self, attr):
        # https://stackoverflow.com/questions/2405590/how-do-i-override-getattr-in-python-without-breaking-the-default-behavior
        # "__getattr__ is only called as a last resort i.e. if there are no attributes in the instance that match the name.
        # For instance, if you access foo.bar, then __getattr__ will only be called if foo has no attribute called bar."
        return getattr(self.terminal, attr)

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            message = line.rstrip()
            if len(message) > 0:
                self.logger.info(message)

    def flush(self):
        pass


def redirect_stdout():
    # Capture print()
    root_logger = logging.getLogger()
    handlers = []
    for handler in root_logger.handlers:
        if handler.name == "console" or handler.name == "file":
            handlers.append(handler)
    sys.stdout = StdoutLogger(root_logger, handlers)


def log_tqdm():
    from src.pl.callbacks.progress import tqdm_logger

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if handler.name == "file":
            tqdm_file_handler = logging.FileHandler(handler.baseFilename)
            tqdm_file_handler.set_name("tqdm_file")
            tqdm_file_handler.setFormatter(logging.Formatter("%(message)s"))
            tqdm_logger.addHandler(tqdm_file_handler)


def fix_logging():
    stdout_filter_errors()
    write_uncaught_exceptions()
    # redirect_stdout()  # Ruins profiling output, so turn it off if the profiler is on.
    # + ruins logging entirely when wandb.require("service") is used, so needs to be called after
    # the logger's initialization.
    log_tqdm()
