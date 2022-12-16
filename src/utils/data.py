from torch.utils.data import Dataset, random_split


def split_train_val(dataset: Dataset, ratio=0.5):
    assert ratio > 0 and ratio < 1

    len1 = round(len(dataset) * ratio)
    len2 = len(dataset) - len1

    return random_split(dataset, [len1, len2])
