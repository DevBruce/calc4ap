from .libs.classifiers import classify_labels, classify_preds
from .libs.calc_ap import CalcAP


__all__ = ['CalcVOCmAP']


class CalcVOCmAP:
    def __init__(self, labels, preds, iou_thr=0.5, conf_thr=0.0):
        self._labels = classify_labels(labels)
        self._preds = classify_preds(preds)
        self._n_classes = len(self._labels)
        self._iou_thr = iou_thr
        self._conf_thr = conf_thr
        self._APs = self._calc_APs()

        mAP_data = self._calc_mAP()
        self._mAP = mAP_data['mAP']
        self._w_mAP = mAP_data['w_mAP']
        self._total_TP = mAP_data['total_TP']
        self._total_FP = mAP_data['total_FP']
        self._total_FN = mAP_data['total_FN']
        self._m_tp_avg_iou = mAP_data['m_tp_avg_iou']
        self._m_precision = mAP_data['m_precision']
        self._m_recall = mAP_data['m_recall']
        self._m_f1_score = mAP_data['m_f1_score']

    def get(self):
        ret = {
            'total_TP': self._total_TP,
            'total_FP': self._total_FP,
            'total_FN': self._total_FN,
            'm_tp_avg_iou': self._m_tp_avg_iou,
            'm_precision': self._m_precision,
            'm_recall': self._m_recall,
            'm_f1_score': self._m_f1_score,
            'APs': self._APs,
            'mAP': self._mAP,
            'w_mAP': self._w_mAP,
        }
        return ret

    def get_summary(self):
        summary = dict()
        for cls_name in self._APs:
            summary[cls_name] = self._APs[cls_name]['AP']
        summary['mAP'] = self._mAP
        return summary

    def _calc_APs(self):
        APs = dict()
        for cls_name in self._labels:
            AP = CalcAP(
                labels=self._labels[cls_name],
                preds=self._preds[cls_name],
                iou_thr=self._iou_thr,
                conf_thr=self._conf_thr,
            ).get()
            APs[cls_name] = AP
        return APs

    def _calc_mAP(self):
        APs = 0.0
        TPs, FPs, FNs = 0, 0, 0
        sum_tp_avg_iou = 0.0
        sum_n_labels = 0
        for cls_name in self._labels:
            APs += self._APs[cls_name]['AP']
            TPs += self._APs[cls_name]['TP']
            FPs += self._APs[cls_name]['FP']
            FNs += self._APs[cls_name]['FN']
            sum_n_labels += self._APs[cls_name]['n_labels']
            sum_tp_avg_iou += self._APs[cls_name]['tp_avg_iou']

        mAP = APs / self._n_classes
        m_tp_avg_iou = sum_tp_avg_iou / self._n_classes
        m_precision = TPs / (TPs + FPs) if (TPs + FPs) else 0.0
        m_recall = TPs / sum_n_labels
        if (m_precision + m_recall) == 0:
            m_f1_score = 0.0
        else:
            m_f1_score = 2 * (m_precision * m_recall) / (m_precision + m_recall)
        
        # Weighted mAP
        w_APs = list()
        for cls_name in self._labels:
            w_AP = self._APs[cls_name]['AP'] * self._APs[cls_name]['n_labels']
            w_APs.append(w_AP)
        w_mAP = sum(w_APs) / sum_n_labels

        ret = {
            'mAP': mAP,
            'w_mAP': w_mAP,
            'total_TP': TPs,
            'total_FP': FPs,
            'total_FN': FNs,
            'm_tp_avg_iou': m_tp_avg_iou,
            'm_precision': m_precision,
            'm_recall': m_recall,
            'm_f1_score': m_f1_score,
        }
        return ret
