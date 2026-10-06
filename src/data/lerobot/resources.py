"""Reuse read-only inputs only during a dataset construction session."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_resources = ContextVar("lerobot_resources", default=None)


@contextmanager
def dataset_resources():
    if _resources.get() is not None:
        yield
        return
    token = _resources.set({})
    try:
        yield
    finally:
        _resources.reset(token)


def reuse_during_construction(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with dataset_resources():
            return function(*args, **kwargs)
    return wrapped


def shared_resource(key, factory, *, refresh=False):
    resources = _resources.get()
    if resources is None:
        return factory()
    if refresh or key not in resources:
        resources[key] = factory()
    return resources[key]
