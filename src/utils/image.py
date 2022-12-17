def read_image(path: str):
    from PIL import Image
    from torchvision import transforms

    if path.startswith("http"):
        import requests

        im = Image.open(requests.get(path, stream=True).raw).convert("RGB")
    else:
        im = Image.open(path).convert("RGB")

    return transforms.ToTensor()(im)
