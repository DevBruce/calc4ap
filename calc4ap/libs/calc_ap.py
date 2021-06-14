from .calc_iou import box_iou


__all__ = ['CalcAP']


class CalcAP:
    def __init__(self, labels, preds, iou_thr=0.5, conf_thr=0.0):
        self._labels = labels
        self._preds = preds
        self._iou_thr = iou_thr
        self._conf_thr = conf_thr
        self._precisions = dict()
        self._recalls = dict()

        pr_data = self._calc_pr()
        self._precisions['origin'] = pr_data['precisions']
        self._recalls['origin'] = pr_data['recalls']
        self._TP = pr_data['TP']
        self._FP = pr_data['FP']
        self._FN = len(self._labels) - self._TP
        self._tp_avg_iou = pr_data['tp_avg_iou']
        self._precision = self._TP / (self._TP + self._FP) if (self._TP + self._FP) else 0.0
        self._recall = self._TP / len(self._labels)
        if (self._precision + self._recall) == 0:
            self._f1_score = 0.0
        else:
            self._f1_score = 2 * (self._precision * self._recall) / (self._precision + self._recall)
        
        voc_ap_data = self._calc_voc_ap()
        self._AP = voc_ap_data['AP']
        self._precisions['interpolated'] = voc_ap_data['interpolated_precisions']
        self._recalls['interpolated'] = voc_ap_data['interpolated_recalls']

    def get(self):
        ret = {
            'n_labels': len(self._labels),
            'TP': self._TP,
            'FP': self._FP,
            'FN': self._FN,
            'tp_avg_iou': self._tp_avg_iou,
            'precision': self._precision,
            'recall': self._recall,
            'f1_score': self._f1_score,
            'precisions': self._precisions,
            'recalls': self._recalls,
            'AP': self._AP,
        }
        return ret
        
    def _calc_ious_confs_useds(self):
        labels_mapped = self._labels.map_by_img_id_with_used()
        ious, confs, useds = list(), list(), list()
        for pred in self._preds:
            *pts, conf, img_id = pred

            img_labels = labels_mapped.get(img_id, [])
            img_ious = list()
            for img_label in img_labels:
                label_pts, used = img_label.get('points'), img_label.get('used')
                iou = box_iou(pts, label_pts)
                img_ious.append(iou)

            if img_ious:
                max_iou = max(img_ious)
                mapped_label_idx = img_ious.index(max_iou)
                used = img_labels[mapped_label_idx].get('used')
                if max_iou >= self._iou_thr and not used:
                    img_labels[mapped_label_idx]['used'] = True
            else:
                max_iou, used = 0.0, False

            ious.append(max_iou)
            confs.append(conf)
            useds.append(used)
        return ious, confs, useds

    def _calc_pr(self):
        TP, FP = 0, 0
        precisions, recalls = [0.0], [0.0]
        tp_ious_sum = 0

        ious, confs, useds = self._calc_ious_confs_useds()
        for iou, conf, used in zip(ious, confs, useds):
            if iou >= self._iou_thr and conf >= self._conf_thr and not used:
                TP += 1
                tp_ious_sum += iou
            else:
                FP += 1
                
            precision = TP / (TP + FP)
            recall = TP / len(self._labels)
            precisions.append(precision)
            recalls.append(recall)
            
        precisions.append(0.0)
        recalls.append(1.0)
        tp_avg_iou = (tp_ious_sum / TP) if TP else 0.0

        ret = {
            'precisions': precisions,
            'recalls': recalls,
            'TP': TP,
            'FP': FP,
            'tp_avg_iou': tp_avg_iou,
        }
        return ret

    def _calc_voc_ap(self):
        precisions = self._precisions['origin'].copy()
        recalls = self._recalls['origin'].copy()
        
        # Interpolation
        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        recalls_changed_idx = list()
        for idx in range(1, len(recalls)):
            if recalls[idx] != recalls[idx-1]:
                recalls_changed_idx.append(idx)

        ap = 0.0
        for idx in recalls_changed_idx:
            recalls_diff = recalls[idx] - recalls[idx-1]
            ap += (recalls_diff * precisions[idx])

        ret = {
            'AP': ap,
            'interpolated_precisions': precisions,
            'interpolated_recalls': recalls,
        }
        return ret
