import os
from redis import Redis
from pottery import Redlock
from contextlib import contextmanager
from filelock import FileLock
import random

HOST = os.environ.get("REDIS_LOCK_HOST")
PORT = os.environ.get("REDIS_LOCK_PORT")
PASSWORD = os.environ.get("REDIS_LOCK_PASSWORD")


@contextmanager
def redis_lock(name, local=False):
    if local or (HOST is None or PORT is None or PASSWORD is None):
        with FileLock(f"{name}.lock") as lock:
            yield
    else:
        redis = Redis(host=HOST, port=PORT, password=PASSWORD)
        lock = Redlock(key=name, masters={redis})
        lock.acquire(timeout=30 + random.randint(0, 10))
        if lock.locked():
            yield
            try:
                lock.release()
            except Exception as e:
                pass
            redis.close()
        else:
            redis.close()
            with redis_lock(name, local=True):
                yield
