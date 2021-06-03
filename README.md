# calc4ap

> Pascal VOC (<http://host.robots.ox.ac.uk/pascal/VOC/>)

<br><br>

## How to Use

```python
from calc4ap.voc import CalcVOCmAP


# (some codes ...)
voc_ap = CalcVOCmAP(labels=labels, preds=preds, iou_thr=0.5, conf_thr=0.0)
ap_summary = voc_ap.get_summary()

car_AP = ap_summary['car']
mAP = ap_summary['mAP']
```

### `CalcVOCmAP` Args

- `preds`: (list): list of `[left, top, right, bottom, confidence, class_name, image_id]`
- `labels`: (list): list of `[left, top, right, bottom, class_name, image_id]`
- `iou_thr`: (float): IoU Threshold (Default: 0.5)
- `conf_thr`: (float): Confidence Threshold (Default: 0.0)
