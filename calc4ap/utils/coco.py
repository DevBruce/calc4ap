import json


__all__ = ['get_labels_from_coco_ann']


def get_labels_from_coco_ann(coco_ann_path):
    coco_ann = json.load(open(coco_ann_path))
    cats = coco_ann['categories']
    anns = coco_ann['annotations']

    cls_map = dict()
    for cat in cats:
        cls_num = cat['id']
        cls_name = cat['name']
        cls_map[cls_num] = cls_name

    labels = list()
    for ann in anns:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        cls_name = cls_map[cat_id]

        x, y, w, h = ann.get('bbox')
        left, top, right, bottom = x, y, x+w, y+h
        labels.append([left, top, right, bottom, cls_name, img_id])
    return labels
