from importlib import import_module

BACKBONE_MODULES = {
    "wan22": ".wan22.loader",
    "cosmos25": ".cosmos25.loader",
    "flux2": ".flux2.loader",
}


def get_backbone_backend(name: str):
    try:
        module = BACKBONE_MODULES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone name: {name!r}") from exc
    return import_module(module, package=__package__)
