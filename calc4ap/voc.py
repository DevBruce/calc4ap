from .libs.classifiers import classify_labels, classify_preds
from .libs.calc_ap import CalcAP


__all__ = ['CalcVOCmAP']


class CalcVOCmAP:
    def __init__(self, labels, preds, iou_thr=0.5, conf_thr=0.0):
        self.labels = classify_labels(labels)
        self.preds = classify_preds(preds)
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.APs = self._get_APs()

        mAP_data = self._get_mAP()
        self.mAP = mAP_data['mAP']
        self.w_mAP = mAP_data['w_mAP']
        self.total_TP = mAP_data['total_TP']
        self.total_FP = mAP_data['total_FP']
        self.total_FN = mAP_data['total_FN']
        self.m_tp_avg_iou = mAP_data['m_tp_avg_iou']
        self.m_precision = mAP_data['m_precision']
        self.m_recall = mAP_data['m_recall']
        self.m_f1_score = mAP_data['m_f1_score']

    def _get_APs(self):
        APs = dict()
        for cls_name in self.labels:
            AP = CalcAP(
                labels=self.labels[cls_name],
                preds=self.preds[cls_name],
                iou_thr=self.iou_thr,
                conf_thr=self.conf_thr,
            )
            APs[cls_name] = AP
        return APs

    def _get_mAP(self):
        APs = 0.0
        TPs, FPs, FNs = 0, 0, 0
        sum_tp_avg_iou = 0.0
        sum_label_length = 0
        for cls_name in self.labels:
            APs += self.APs[cls_name].AP
            TPs += self.APs[cls_name].TP
            FPs += self.APs[cls_name].FP
            FNs += self.APs[cls_name].FN
            sum_label_length += len(self.APs[cls_name].labels)
            sum_tp_avg_iou += self.APs[cls_name].tp_avg_iou

        mAP = APs / len(self.labels)
        m_tp_avg_iou = sum_tp_avg_iou / len(self.labels)
        m_precision = TPs / (TPs + FPs) if (TPs + FPs) else 0.0
        m_recall = TPs / sum_label_length
        if (m_precision + m_recall) == 0:
            m_f1_score = 0.0
        else:
            m_f1_score = 2 * (m_precision * m_recall) / (m_precision + m_recall)
        
        # Weighted mAP
        w_APs = list()
        for cls_name in self.labels:
            w_AP = self.APs[cls_name].AP * len(self.APs[cls_name].labels)
            w_APs.append(w_AP)
        w_mAP = sum(w_APs) / sum_label_length

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

    def get_summary(self):
        summary_data = dict()
        for cls_name in self.APs:
            summary_data[cls_name] = self.APs[cls_name].AP
        summary_data['mAP'] = self.mAP
        return summary_data
