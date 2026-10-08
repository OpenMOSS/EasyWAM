from functools import wraps

import torch


def _key(value):
    if isinstance(value, torch.Tensor):
        version = None if value.is_inference() else value._version
        return (id(value), version, value.device, value.dtype, tuple(value.shape))
    if isinstance(value, (tuple, list)):
        return tuple(_key(item) for item in value)
    return value


def cache_static_method(*attributes):
    def decorate(method):
        cache_name = "_static_cache_" + method.__name__

        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if torch.compiler.is_compiling():
                return method(self, *args, **kwargs)
            settings = []
            for path in attributes:
                value = self
                for name in path.split("."):
                    value = getattr(value, name)
                settings.append(_key(value))
            key = (args, tuple(sorted(kwargs.items())), tuple(settings))
            cached = getattr(self, cache_name, None)
            if cached is not None and cached[0] == key:
                return cached[1]
            result = method(self, *args, **kwargs)
            setattr(self, cache_name, (key, result))
            return result

        return wrapped
    return decorate
