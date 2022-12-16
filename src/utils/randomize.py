import torch


def random_choice(
    input: torch.Tensor, num_samples: int, replacement=False, auto_replace=False
):
    """
    Args:
        input (Tensor): Shape: (N, ...)
        num_samples (int): Số lượng mẫu cần lấy
        replacement (boolean): Cho phép lấy mẫu có hoàn lại
        auto_replace (boolean): Tự động hoàn lại khi số lượng mẫu cần lấy lớn hơn số lượng mẫu có sẵn

    Returns:
        ids (Tensor): Các chỉ số của các mẫu được lấy. Shape: (num_samples)
    """

    if auto_replace:
        replacement = input.size(0) < num_samples

    dtype = torch.int64
    device = input.device

    if replacement:
        ids = torch.randint(
            input.size(0), size=(num_samples,), dtype=dtype, device=device
        )
    else:
        ids = torch.randperm(input.size(0), dtype=dtype, device=device)[:num_samples]

    return ids
