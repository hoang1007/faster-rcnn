import matplotlib.pyplot as plt
from matplotlib import patches


def visualize(
    img,
    boxes,
    label_ids,
    box_scores,
    idx2classes,
    confident_thresh=0.5,
    edgecolors=None,
):
    if edgecolors is None:
        edgecolors = ["red", "white", "orange", "blue"]
    elif not isinstance(edgecolors, list):
        raise ValueError

    fig, axis = plt.subplots(1)

    for i, box in enumerate(boxes):
        label = idx2classes[label_ids[i]]
        score = round(box_scores[i].item(), 2)

        if score >= confident_thresh:
            xtl = box[0]
            ytl = box[1]
            w = box[2] - box[0] + 1  # width
            h = box[3] - box[1] + 1  # height

            color = edgecolors[i % len(edgecolors)]

            rect = patches.Rectangle(
                (xtl, ytl), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
            axis.add_patch(rect)
            axis.annotate(
                f"{label} ({score})",
                (xtl, ytl + 1),
                color=color,
                weight="bold",
                fontsize=10,
                ha="left",
                va="bottom",
            )

    plt.imshow(img.permute(1, 2, 0))
    plt.show()
