import math
import torch


def get_box_info(boxes: torch.Tensor):
    """
    Args:
        boxes (Tensor): các hộp ở định dạng `(x_tl, y_tl, x_br, y_br)`. Shape (N, 4)

    Returns:
        xctrs: Tọa độ trung tâm của hộp theo trục x. Shape (N,)
        yctrs: Tọa độ trung tâm của hộp theo trục y. Shape (N,)
        ws: Chiều dài của hộp. Shape (N,)
        hs: Chiều cao của hộp. Shape (N,)
    """

    assert len(boxes.shape) == 2 and boxes.size(1) == 4

    ws = boxes[:, 2] - boxes[:, 0] + 1  # x_br - x_tl + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1  # y_br - y_tl + 1

    xctrs = boxes[:, 0] + 0.5 * (ws - 1)  # x_tl + (w - 1) / 2
    yctrs = boxes[:, 1] + 0.5 * (hs - 1)  # y_tl + (h - 1) / 2

    return xctrs, yctrs, ws, hs


def box_info_to_boxes(xctrs, yctrs, ws, hs):
    """
    Args:
        xctrs: Tọa độ trung tâm của hộp theo trục x. Shape (N,)
        yctrs: Tọa độ trung tâm của hộp theo trục y. Shape (N,)
        ws: Chiều dài của hộp. Shape (N,)
        hs: Chiều cao của hộp. Shape (N,)

    Returns:
        boxes (Tensor): các hộp ở định dạng `(x_tl, y_tl, x_br, y_br)`. Shape (N, 4)
    """

    assert len(xctrs.shape) == 1
    assert len(yctrs.shape) == 1
    assert len(ws.shape) == 1
    assert len(hs.shape) == 1

    delta_x = 0.5 * (ws - 1)
    delta_y = 0.5 * (hs - 1)

    x_tls = xctrs - delta_x
    y_tls = yctrs - delta_y
    x_brs = xctrs + delta_x
    y_brs = yctrs + delta_y

    boxes = torch.stack((x_tls, y_tls, x_brs, y_brs), dim=1)

    assert len(boxes.shape) == 2 and boxes.size(1) == 4

    return boxes


def bbox_transform(ex_boxes, tar_boxes):
    """
    Args:
        ex_boxes: Hộp nguồn ở định dạng (x_tl, y_tl, x_br, y_br). Shape (N, 4)
        tar_boxes: Hộp đích ở định dạng (x_tl, y_tl, x_br, y_br). Shape (N, 4)

    Returns:
        deltas: Shape (N, 4)
    """

    ex_xctrs, ex_yctrs, ex_ws, ex_hs = get_box_info(ex_boxes)
    tar_xctrs, tar_yctrs, tar_ws, tar_hs = get_box_info(tar_boxes)

    tx = (tar_xctrs - ex_xctrs) / ex_ws
    ty = (tar_yctrs - ex_yctrs) / ex_hs
    tw = torch.log(tar_ws / ex_ws)
    th = torch.log(tar_hs / ex_hs)

    deltas = torch.stack((tx, ty, tw, th), dim=1)

    return deltas


def bbox_transform_inv(
    boxes: torch.Tensor, deltas: torch.Tensor, clamp_thresh=math.log(1000 / 16)
):
    """
    Args:
        boxes: Shape (N, 4)
        deltas: Shape (N, 4)

    Returns:
        tar_boxes: Hộp dích ở định dạng (x_tl, y_tl, x_br, y_br). Shape (N, 4)
    """

    xctrs, yctrs, ws, hs = get_box_info(boxes)

    # tx, ty, tw, th = deltas.t()
    tx = deltas[:, 0]
    ty = deltas[:, 1]
    tw = deltas[:, 2]
    th = deltas[:, 3]

    tw = torch.clamp(tw, max=clamp_thresh)
    th = torch.clamp(th, max=clamp_thresh)

    tar_xctrs = tx * ws + xctrs
    tar_yctrs = ty * hs + yctrs
    tar_ws = torch.exp(tw) * ws
    tar_hs = torch.exp(th) * hs

    tar_boxes = box_info_to_boxes(tar_xctrs, tar_yctrs, tar_ws, tar_hs)

    return tar_boxes


def bbox_iou(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (
        query_boxes[:, 3] - query_boxes[:, 1] + 1
    )

    iw = (
        torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t())
        - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())
        + 1
    ).clamp(min=0)
    ih = (
        torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t())
        - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t())
        + 1
    ).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return overlaps


def clip_boxes(boxes: torch.Tensor, height, width):
    """
    Đưa các hộp về trong khoảng kích thước đã cho.

    Args:
        boxes: Các hộp ở định dạng (x_tl, y_tl, x_br, y_br). Shape (N, 4)
    """

    boxes = torch.stack(
        (
            boxes[:, 0].clamp(0, width - 1),
            boxes[:, 1].clamp(0, height - 1),
            boxes[:, 2].clamp(0, width - 1),
            boxes[:, 3].clamp(0, height - 1),
        ),
        dim=1,
    )

    return boxes
