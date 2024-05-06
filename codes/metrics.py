from scipy import stats
from sklearn import metrics
import numpy as np
import torch
import os
import json
import pandas as pd
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred


def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def get_evaluate_metrics(multi_label=True):
    if multi_label:
        def evaluate(output, target, **kwargs):
            AP = metrics.average_precision_score(target, output, average=None)
            return {'mAP': np.mean(AP)}
            # mAP = ASL_mAP(output, target)
            # return {'mAP': mAP}
        return evaluate
    else:
        def evaluate(output, target, **kwargs):
            (top1, _), _ = accuracy(output, target, **kwargs)
            return {'acc': top1}
        return evaluate




class AccuracyPerClass(Metric):
    """
    Compute accuracy for each class
    input: preds, targets
    """
    def __init__(self, n_class=309, output_transform=lambda x: x, device="cpu"):
        self.n_class = n_class
        self._num_correct_per_class = [0] * n_class
        self._num_examples_per_class = [1e-12] * n_class
        super(AccuracyPerClass, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct_per_class = [0] * self.n_class
        self._num_examples_per_class = [1e-12] * self.n_class
        super(AccuracyPerClass, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        preds = torch.argmax(y_pred, dim=1)
        correct = torch.eq(preds, y).view(-1)
        for idx, gt in enumerate(y):
            self._num_correct_per_class[gt] += correct[idx]
            self._num_examples_per_class[gt] += 1


    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        acc_per_class = torch.tensor(self._num_correct_per_class) / torch.tensor(self._num_examples_per_class)
        return acc_per_class

def evaluate_per_class(preds, targets, output_path, mapping=None):
    preds, targets = preds.numpy(), targets.numpy()
    n_targets = np.max(targets) + 1
    preds = np.argmax(preds, axis=1)
    pred_per_target = np.zeros((n_targets, n_targets), dtype=np.int32)
    for target in range(n_targets):
        indices = np.where(targets == target)[0]
        for idx in preds[indices]:
            pred_per_target[target][idx] += 1
    if mapping:
        with open(mapping, 'r') as input:
            data = json.load(input)
        reverse = {v: k for k, v in data.items()}
        target_names = [reverse[i] for i in range(n_targets)]
        df = pd.DataFrame(pred_per_target, index=target_names, columns=target_names)
        df.to_csv(os.path.join(output_path, 'pred_per_target.csv'))
    else:
        np.save(os.path.join(
            output_path, 'pred_per_target.npy'), pred_per_target)

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    

def get_confusion(preds, targets, mapping):
    preds = np.argmax(preds.numpy(), axis=1)
    targets = targets.numpy()
    labels = [mapping[i] for i in range(len(mapping))]
    confusion_matrix = metrics.confusion_matrix(
        targets, preds, labels=range(len(mapping)), normalize='true')
    return confusion_matrix, labels





"""
Adapted from https://github.com/JunwenBai/c-gmvae/blob/master/evals.py
Metrics contains:
    - Hamming Accuracy
    - example-F1, micro-F1, macro-F1,
    - precision@1
"""

from sklearn import metrics
import math
import os
from copy import deepcopy
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


def get_mlc_metrics(preds, targets):
    # THRESHOLDS = [i / 10. for i in range(10)]
    THRESHOLDS = [i / 10. for i in range(1, 10)]
    METRICS = ['HA', 'ebF1', 'miF1', 'maF1', 'p_at_1']

    best_test_metrics = None
    for threshold in THRESHOLDS:
        test_metrics = compute_metrics(preds, targets, threshold, all_metrics=False)
        if best_test_metrics == None:
            best_test_metrics = {}
            for metric in METRICS:
                best_test_metrics[metric] = test_metrics[metric]
        else:
            for metric in METRICS:
                if 'FDR' in metric:
                    best_test_metrics[metric] = min(best_test_metrics[metric], test_metrics[metric])
                else:
                    # old = best_test_metrics[metric]
                    best_test_metrics[metric] = max(best_test_metrics[metric], test_metrics[metric])
                    # if old != best_test_metrics[metric]:
                        # print(threshold, metric, best_test_metrics[metric])
    return best_test_metrics


def ranking_precision_score(Y_true, Y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    sum_prec = 0.
    n = len(Y_true)

    unique_Y = np.unique(Y_true)

    if len(unique_Y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_Y[1]
    n_pos = np.sum(Y_true == pos_label, axis=1)
    order = np.argsort(Y_score, axis=1)[:, ::-1]
    Y_true = np.array([x[y] for x, y in zip(Y_true, order[:, :k])])

    n_relevant = np.sum(Y_true == pos_label, axis=1)

    cnt = k
    prec = np.divide(n_relevant.astype(float), cnt)
    return np.average(prec)


def subset_accuracy(true_targets, predictions, per_sample=False, axis=0):
    result = np.all(true_targets == predictions, axis=axis)
    if not per_sample:
        result = np.mean(result)
    return result


def hamming_loss(true_targets, predictions, per_sample=False, axis=0):
    result = np.mean(np.logical_xor(true_targets, predictions), axis=axis)
    if not per_sample:
        result = np.mean(result)
    return result


def compute_tp_fp_fn(true_targets, predictions, axis=0):
    tp = np.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = np.sum(np.logical_not(true_targets) * predictions, 
                   axis=axis).astype('float32')
    fn = np.sum(true_targets * np.logical_not(predictions), 
                   axis=axis).astype('float32')
    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=0):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)

    numerator = 2*tp
    denominator = (np.sum(true_targets, axis=axis).astype('float32') + np.sum(predictions, axis=axis).astype('float32'))

    zeros = np.where(denominator == 0)[0]

    denominator = np.delete(denominator, zeros)
    numerator = np.delete(numerator, zeros)

    example_f1 = numerator/denominator

    if per_sample:
        f1 = example_f1
    else:
        f1 = np.mean(example_f1)

    return f1


def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*np.sum(tp) / \
            float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
            return c[np.isfinite(c)]
        tmp = safe_div(2*tp, 2*tp + fp + fn + 1e-6)
        #print(tmp)
        f1 = np.mean(safe_div(2*tp, 2*tp + fp + fn + 1e-6))

    return f1


def f1_score(true_targets, predictions, average='micro', axis=0):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)
    return f1


def compute_fdr(all_targets, all_predictions, fdr_cutoff=0.5):
    fdr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:, i], all_predictions[:, i], pos_label=1)
            fdr = 1- precision
            cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
            fdr_at_cutoff = recall[cutoff_index]
            if not math.isnan(fdr_at_cutoff):
                fdr_array.append(np.nan_to_num(fdr_at_cutoff))
        except: 
            pass
    
    fdr_array = np.array(fdr_array)
    mean_fdr = np.mean(fdr_array)
    median_fdr = np.median(fdr_array)
    var_fdr = np.var(fdr_array)
    return mean_fdr, median_fdr, var_fdr, fdr_array


def compute_aupr(all_targets, all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:, i], all_predictions[:, i], pos_label=1)
        auPR = metrics.auc(recall, precision)
        if not math.isnan(auPR):
            aupr_array.append(np.nan_to_num(auPR))
    aupr_array = np.array(aupr_array)
    mean_aupr = np.mean(aupr_array)
    median_aupr = np.median(aupr_array)
    var_aupr = np.var(aupr_array)
    return mean_aupr, median_aupr, var_aupr, aupr_array


def compute_auc(all_targets, all_predictions):
    auc_array = []
    for i in range(all_targets.shape[1]):
        try:  
            auROC = metrics.roc_auc_score(all_targets[:, i], all_predictions[:, i])
            auc_array.append(auROC)
        except ValueError:
            pass
    auc_array = np.array(auc_array)
    mean_auc = np.mean(auc_array)
    median_auc = np.median(auc_array)
    var_auc = np.var(auc_array)
    return mean_auc, median_auc, var_auc, auc_array


def compute_metrics(predictions, targets, threshold, all_metrics=True):
    all_targets = deepcopy(targets)
    all_predictions = deepcopy(predictions)

    if all_metrics:
        meanAUC, medianAUC, varAUC, allAUC = compute_auc(all_targets, all_predictions)
        meanAUPR, medianAUPR, varAUPR, allAUPR = compute_aupr(all_targets, all_predictions)
        meanFDR, medianFDR, varFDR, allFDR = compute_fdr(all_targets, all_predictions)
    else:
        meanAUC, medianAUC, varAUC, allAUC = 0, 0, 0, 0
        meanAUPR, medianAUPR, varAUPR, allAUPR = 0, 0, 0, 0
        meanFDR, medianFDR, varFDR, allFDR = 0, 0, 0, 0

    p_at_1 = 0.
    p_at_3 = 0.
    p_at_5 = 0.
    
    p_at_1 = ranking_precision_score(Y_true=all_targets, Y_score=all_predictions, k=1)
    p_at_3 = ranking_precision_score(Y_true=all_targets, Y_score=all_predictions, k=3)
    p_at_5 = ranking_precision_score(Y_true=all_targets, Y_score=all_predictions, k=5)
    
    optimal_threshold = threshold
    
    all_predictions[all_predictions < optimal_threshold] = 0
    all_predictions[all_predictions >= optimal_threshold] = 1

    
    acc_ = list(subset_accuracy(all_targets, all_predictions, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions, axis=1, per_sample=True))        
    ACC = np.mean(acc_)
    hl = np.mean(hl_)
    HA = 1 - hl
    ebF1 = np.mean(exf1_)
    tp, fp, fn = compute_tp_fp_fn(all_targets, all_predictions, axis=0)

    miF1 = f1_score_from_stats(tp, fp, fn, average='micro')
    maF1 = f1_score_from_stats(tp, fp, fn, average='macro')

    metrics_dict = {}
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['miF1'] = miF1
    metrics_dict['maF1'] = maF1
    metrics_dict['meanAUC'] = meanAUC
    metrics_dict['medianAUC'] = medianAUC
    metrics_dict['varAUC'] = varAUC
    metrics_dict['allAUC'] = allAUC
    metrics_dict['meanAUPR'] = meanAUPR
    metrics_dict['medianAUPR'] = medianAUPR
    metrics_dict['varAUPR'] = varAUPR
    metrics_dict['allAUPR'] = allAUPR
    metrics_dict['meanFDR'] = meanFDR
    metrics_dict['medianFDR'] = medianFDR
    metrics_dict['varFDR'] = varFDR
    metrics_dict['allFDR'] = allFDR
    metrics_dict['p_at_1'] = p_at_1
    metrics_dict['p_at_3'] = p_at_3
    metrics_dict['p_at_5'] = p_at_5

    return metrics_dict




def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(predicted, actual):
    
    predicted = predicted.numpy()
    actual = actual.numpy()
    gt_label = actual.astype(np.int32)
    num = gt_label.shape[-1]

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = predicted[:,class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    return mAP




def MCAR_mAP(outputs, targets):
    ap = torch.zeros(outputs.shape[1])
    rg = torch.arange(1, outputs.shape[0]).float()
    # compute average precision for each class
    for k in range(outputs.shape[1]):
        # sort scores
        _scores = outputs[:, k]
        _targets = targets[:, k]
        # compute average precision
        ap[k] = _average_precision(_scores, _targets)
    return (100 * ap).mean()

def _average_precision(outputs, targets, difficult_examples=True):
    # sort examples
    _, indices = torch.sort(outputs, dim=0, descending=True)
    # Computes prec@i
    pos_count = 0.
    total_count = 0.
    precision_at_i = 0.
    for i in indices:
        label = targets[i]
        if difficult_examples and label == 0:
            continue
        if label == 1:
            pos_count += 1
        total_count += 1
        if label == 1:
            precision_at_i += pos_count / total_count
    if pos_count == 0:
        precision_at_i = 0
    else:
        precision_at_i /= pos_count
    return precision_at_i



def ASL_average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def ASL_mAP(preds, targs):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = ASL_average_precision(scores, targets)
    return 100 * ap.mean()