__all__ = ['box_iou']


def box_iou(box1_pts, box2_pts):
    """Calculate IoU of box1 and box2

    Args:
        box1_pts (list): [left, top, right, bottom] (xyrb)
        box2_pts (list): [left, top, right, bottom] (xyrb)

    Returns:
        float: IoU of box1 and box2
    """
    box_intersection_pts = [
        max(box1_pts[0], box2_pts[0]),
        max(box1_pts[1], box2_pts[1]),
        min(box1_pts[2], box2_pts[2]),
        min(box1_pts[3], box2_pts[3]),
        ]
    intersection_width = box_intersection_pts[2] - box_intersection_pts[0] + 1
    intersection_height = box_intersection_pts[3] - box_intersection_pts[1] + 1
    intersection_area = intersection_width * intersection_height
    
    if intersection_width > 0 and intersection_height > 0:
        box1_width = box1_pts[2] - box1_pts[0] + 1
        box1_height = box1_pts[3] - box1_pts[1] + 1
        box2_width = box2_pts[2] - box2_pts[0] + 1
        box2_height = box2_pts[3] - box2_pts[1] + 1
        
        box1_area = box1_width * box1_height
        box2_area = box2_width * box2_height
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area
    else:
        iou = 0.0
    return iou
