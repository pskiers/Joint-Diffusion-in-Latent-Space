from torchvision import transforms


def get_torchvision_transform(name: str, kwargs: dict):
    try:
        cls_object = getattr(transforms, name)
    except AttributeError:
        raise AttributeError(f"No {name} transform found in torchvision")
    return cls_object(**kwargs)
