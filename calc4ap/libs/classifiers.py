from collections import defaultdict


__all__ = ['classify_labels', 'classify_preds']


def classify_labels(labels):
    """
    Args:
        labels (list): list of [left, top, right, bottom, class_name, image_id]

    Returns:
        defaultdict: {key: class_name (str), value: ClassLabel Object}
    """
    labels_classified = defaultdict(_ClassLabel)
    for label in labels:
        *pts, cls_name, img_id = label
        labels_classified[cls_name].labels.append([*pts, img_id])
    return labels_classified


def classify_preds(preds):
    """
    Args:
        preds (list): list of [left, top, right, bottom, confidence, class_name, image_id]

    Returns:
        dict: {key: class_name (str), value: list of [left, top, right, bottom, confidence, image_id]}
    """
    preds_classified = defaultdict(list)
    for pred in preds:
        *pts, confidence, cls_name, img_id = pred
        preds_classified[cls_name].append([*pts, confidence, img_id])

    # Descending Ordering with Confidence
    for cls_name in preds_classified:
        preds_classified[cls_name].sort(key=lambda x : x[-2], reverse=True)
    return preds_classified


class _ClassLabel:
    def __init__(self):
        self.labels = list()
    
    def __len__(self):
        return len(self.labels)

    def map_by_img_id_with_used(self):
        labels_mapped = defaultdict(list)
        for label in self.labels:
            *pts, img_id = label
            labels_mapped[img_id].append({'points': pts, 'used': False})
        return labels_mapped
