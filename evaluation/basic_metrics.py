import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import numpy as np
import math
import copy
import sklearn
from typing import Callable, Dict, Any, Tuple, Optional, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import time

from math import log, floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'

import multiprocessing as mp
def generate_curve(label, score, slidingWindow, version='opt', thre=250):
    if version =='opt_mem':
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)


    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)

    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d

def inverse_proportional_cardinality_fn(cardinality: int, gt_length: int) -> float:
    r"""
    Cardinality function that assigns an inversely proportional weight to predictions within a single ground-truth
    window.

    This is the default cardinality function recommended in [Tatbul2018]_.

    .. note::
       This function leads to a metric that is not recall-consistent! Please see [Wagner2023]_ for more details.

    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window (unused).
    :return: The cardinality factor :math:`\frac{1}{\text{cardinality}}`.

    .. [Tatbul2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich.
        Precision and recall for time series. Advances in neural information processing systems. 2018;31.
    .. [Wagner2023] D. Wagner, T. Michels, F.C.F. Schulz, A. Nair, M. Rudolph, and M. Kloft.
        TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection.
        Transactions on Machine Learning Research (TMLR), (to appear) 2023.
    """
    return 1 / max(1, cardinality)

def constant_bias_fn(inputs: torch.Tensor) -> float:
    r"""
    Compute the overlap size for a constant bias function that assigns the same weight to all positions.

    This functions computes

    .. math::
        \omega(\text{inputs}) = \frac{1}{n} \sum_{i = 1}^{n} \text{inputs}_i,

    where :math:`n = \lvert \text{inputs} \rvert`.

    .. note::
       To improve the runtime of our algorithm, we calculate the overlap :math:`\omega` directly as part of the bias
       function.

    :param inputs: A 1-D :class:`~torch.Tensor` containing the predictions inside a ground-truth window.
    :return: The overlap :math:`\omega`.
    """
    return torch.sum(inputs).item() / inputs.shape[0]

def improved_cardinality_fn(cardinality: int, gt_length: int):
    r"""
    Recall-consistent cardinality function introduced by [Wagner2023]_ that assigns lower weight to ground-truth windows
    that overlap with many predicted windows.

    This function computes

    .. math::
        \left(\frac{\text{gt_length} - 1}{\text{gt_length}}\right)^{\text{cardinality} - 1}.

    :param cardinality: Number of predicted windows that overlap the ground-truth window in question.
    :param gt_length: Length of the ground-truth window.
    :return: The cardinality factor.
    """
    return ((gt_length - 1) / gt_length) ** (cardinality - 1)

class basic_metricor():
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias
        self.eps = 1e-15

    def detect_model(self, model, label, contamination = 0.1, window = 100, is_A = False, is_threshold = True):
        if is_threshold:
            score = self.scale_threshold(model.decision_scores_, model._mu, model._sigma)
        else:
            score = self.scale_contamination(model.decision_scores_, contamination = contamination)
        if is_A is False:
            scoreX = np.zeros(len(score)+window)
            scoreX[math.ceil(window/2): len(score)+window - math.floor(window/2)] = score
        else:
            scoreX = score

        self.score_=scoreX
        L = self.metric(label, scoreX)
        return L

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1/score

    def b(self, i, length):
        bias = self.bias
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1

    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3*score_sigma)).astype(int)

    def _adjust_predicts(self, score, label, threshold=None, pred=None, calc_latency=False):
        """
        Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

        Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                A point is labeled as "anomaly" if its score is higher than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):

        Returns:
            np.ndarray: predict labels
        """
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)
        label = np.asarray(label)
        latency = 0
        if pred is None:
            predict = score > threshold
        else:
            predict = copy.deepcopy(pred)
        actual = label > 0.1
        anomaly_state = False
        anomaly_count = 0
        for i in range(len(score)):
            if actual[i] and predict[i] and not anomaly_state:
                    anomaly_state = True
                    anomaly_count += 1
                    for j in range(i, 0, -1):
                        if not actual[j]:
                            break
                        else:
                            if not predict[j]:
                                predict[j] = True
                                latency += 1
            elif not actual[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
        if calc_latency:
            return predict, latency / (anomaly_count + 1e-4)
        else:
            return predict

    def adjustment(self, gt, pred):
        adjusted_pred = np.array(pred)
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if adjusted_pred[j] == 0:
                            adjusted_pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if adjusted_pred[j] == 0:
                            adjusted_pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                adjusted_pred[i] = 1
        return adjusted_pred

    def metric_new(self, label, score, preds, plot_ROC=False, alpha=0.2):
        '''input:
               Real labels and anomaly score in prediction

           output:
               AUC,
               Precision,
               Recall,
               F-score,
               Range-precision,
               Range-recall,
               Range-Fscore,
               Precison@k,

            k is chosen to be # of outliers in real labels
        '''
        if np.sum(label) == 0:
            print('All labels are 0. Label must have groud truth value for calculating AUC score.')
            return None

        if np.isnan(score).any() or score is None:
            print('Score must not be none.')
            return None

        #area under curve
        auc = metrics.roc_auc_score(label, score)
        # plor ROC curve
        if plot_ROC:
            fpr, tpr, thresholds  = metrics.roc_curve(label, score)
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            # display.plot()

        #precision, recall, F
        if preds is None:
            preds = score > (np.mean(score)+3*np.std(score))
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        precision = Precision[1]
        recall = Recall[1]
        f = F[1]

        #point-adjust
        adjust_preds = self._adjust_predicts(score, label, pred=preds)
        PointF1PA = metrics.f1_score(label, adjust_preds)

        #range anomaly
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha)
        Rprecision = self.range_recall_new(preds, label, 0)[0]

        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))

        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k

        L = [auc, precision, recall, f, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, Rf, precision_at_k]
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_ROC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)

    def metric_PointF1(self, label, score, preds=None):
        if preds is None:
            precision, recall, thresholds = metrics.precision_recall_curve(label, score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 0.00001)
            F1 = np.max(f1_scores)
            threshold = thresholds[np.argmax(f1_scores)]
        else:
            Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            F1 = F[1]
        return F1

    def metric_standard_F1(self, true_labels, anomaly_scores, threshold=None):
        """
        Calculate F1, Precision, Recall, and other metrics for anomaly detection.

        Args:
            anomaly_scores: np.ndarray, anomaly scores (continuous values)
            true_labels: np.ndarray, ground truth binary labels (0=normal, 1=anomaly)
            threshold: float, optional. If None, will use optimal threshold based on F1 score

        Returns:
            dict: Dictionary containing various metrics
        """
        # If no threshold provided, find optimal threshold
        if threshold is None:
            thresholds = np.linspace(0, 1, 1500)
            best_f1 = 0
            best_threshold = 0

            for t in tqdm(thresholds, total=len(thresholds), desc="Finding optimal threshold"):
                threshold = np.quantile(anomaly_scores, t)
                predictions = (anomaly_scores >= threshold).astype(int)
                if len(np.unique(predictions)) > 1:  # Avoid division by zero
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        true_labels, predictions, average='binary', zero_division=0
                    )
                    # print(f1, t)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            threshold = best_threshold
        # print("aaa", threshold, best_threshold, best_f1)
        # Calculate predictions based on threshold
        predictions = (anomaly_scores >= threshold).astype(int)

        # Calculate basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        # print(threshold, f1)
        return {
            'F1': f1,
            'Recall': recall,
            'Precision': precision, }


    def metric_Affiliation(self, label, score, preds=None):
        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        # Ensure proper data types to avoid float/integer issues
        label = np.asarray(label, dtype=int)
        score = np.asarray(score, dtype=float)

        # Convert ground truth to events once, outside the loop
        events_gt = convert_vector_to_events(label)

        if preds is None:
            # print("Calculating afiliation metrics using score thresholds.")
            p_values = np.linspace(0, 1, 1500)
            # print(f"Using {thresholds} thresholds for affiliation metrics.")
            Affiliation_scores = []
            Affiliation_Precision_scores = []
            Affiliation_Recall_scores = []
            # print("Score values", score)

            for p in tqdm(p_values, total=(len(p_values)), desc="Calculating Affiliation Metrics"):
                threshold = np.quantile(score, p)
                preds_loop = (score > threshold).astype(int)

                events_pred = convert_vector_to_events(preds_loop)
                # events_gt is already calculated
                Trange = (0, len(preds_loop))

                affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

                Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
                Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
                # --- FIX 1: Prevent division by zero ---
                denominator = Affiliation_Precision + Affiliation_Recall
                if denominator > 0:
                    Affiliation_F = 2 * Affiliation_Precision * Affiliation_Recall / (denominator + self.eps)
                else:
                    Affiliation_F = 0.0
                # # Use a local variable for the F1 score in the loop
                # Affiliation_F = 2 * Affiliation_Precision * Affiliation_Recall / (
                #             Affiliation_Precision + Affiliation_Recall + self.eps)

                Affiliation_scores.append(Affiliation_F)
                Affiliation_Precision_scores.append(Affiliation_Precision)
                Affiliation_Recall_scores.append(Affiliation_Recall)

            # Find the best scores after the loop
            # print("Here are the Affiliation scores:", Affiliation_scores)
            best_index = np.argmax(Affiliation_scores)
            # print(f"Best Affiliation F1 score found at index {best_index} with value {Affiliation_scores[best_index]}")
            Best_Affiliation_F1 = Affiliation_scores[best_index]
            Best_Affiliation_Precision = Affiliation_Precision_scores[best_index]
            Best_Affiliation_Recall = Affiliation_Recall_scores[best_index]

        else:
            print("Using provided predictions for affiliation metrics.")
            # This block runs when 'preds' is provided
            events_pred = convert_vector_to_events(preds)
            Trange = (0, len(preds))

            affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

            # FIX: Assign the calculated values to the 'Best_' variables
            # so they exist for the return statement.
            Best_Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
            Best_Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
            Best_Affiliation_F1 = 2 * Best_Affiliation_Precision * Best_Affiliation_Recall / (
                        Best_Affiliation_Precision + Best_Affiliation_Recall + self.eps)

        # FIX: Corrected the typo from Best_Affiliation_Rec to Best_Affiliation_Recall
        return Best_Affiliation_F1, Best_Affiliation_Precision, Best_Affiliation_Recall

    def metric_RF1(self, label, score, preds=None):

        if preds is None:
            q_values = np.linspace(0, 1, 1000)
            Rf1_scores = []
            thresholds = []
            for q in tqdm(q_values, total=(len(q_values)), desc="Calculating RF1 Metrics"):
                # Calculate prediction
                threshold = np.quantile(score, q)
                preds = (score > threshold).astype(int)

                Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
                Rprecision = self.range_recall_new(preds, label, 0)[0]
                if Rprecision + Rrecall==0:
                    Rf=0
                else:
                    Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

                Rf1_scores.append(Rf)
                thresholds.append(threshold)

            RF1_Threshold = thresholds[np.argmax(Rf1_scores)]
            RF1 = max(Rf1_scores)
        else:
            Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
            Rprecision = self.range_recall_new(preds, label, 0)[0]
            if Rprecision + Rrecall==0:
                RF1=0
            else:
                RF1 = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
        return RF1

    # def metric_F1_T(self, labels: torch.Tensor, scores: torch.Tensor):
    #     """
    #     Computes the F1 score for time series anomaly detection by finding the best threshold.
    #
    #     Args:
    #         labels (torch.Tensor): Ground truth labels for the time series data.
    #         scores (torch.Tensor): Anomaly scores predicted by the model.
    #
    #     Returns:
    #         Tuple[float, Dict[str, Any]]: The best F1 score and a dictionary with additional metrics.
    #     """
    #     result = {}
    #     labels = torch.tensor(labels, dtype=torch.int)
    #     score = torch.tensor(scores, dtype=torch.float)
    #     f1, details = self.__best_ts_fbeta_score(labels, score, beta=1,)
    #     result['thre_T'] = details['threshold']
    #     result['ACC_T'] = sklearn.metrics.accuracy_score(labels, score > details['threshold'])
    #     result['P_T'] = details['precision']
    #     result['R_T'] = details['recall']
    #     result['F1_T'] = f1
    #
    #     return result

    def metric_F1_T(self, labels: torch.Tensor, scores: torch.Tensor, use_parallel=True, 
                    parallel_method='chunked', chunk_size=10, max_workers=8):
        """
        Computes the F1 score with optional parallel processing.

        Args:
            labels: Ground truth labels
            scores: Anomaly scores
            use_parallel: Whether to use parallel processing (default: True)
            parallel_method: Type of parallel processing ('standard' or 'chunked')
            chunk_size: Size of chunks for chunked parallel processing
            max_workers: Maximum number of worker threads
        """
        result = {}
        labels = torch.tensor(labels, dtype=torch.int)
        score = torch.tensor(scores, dtype=torch.float)

        # Choose which method to use
        if use_parallel:
            if parallel_method == 'chunked':
                f1, details = self.__best_ts_fbeta_score_parallel_chunked(
                    labels, score, beta=1, chunk_size=chunk_size, max_workers=max_workers
                )
            else:  # standard parallel
                f1, details = self.__best_ts_fbeta_score_parallel(labels, score, beta=1)
        else:
            f1, details = self.__best_ts_fbeta_score(labels, score, beta=1)

        result['thre_T'] = details['threshold']
        result['ACC_T'] = sklearn.metrics.accuracy_score(labels, score > details['threshold'])
        result['P_T'] = details['precision']
        result['R_T'] = details['recall']
        result['F1_T'] = f1

        return result

    def __best_ts_fbeta_score_parallel(self, labels: torch.Tensor, scores: torch.Tensor, beta: float,
                                       recall_cardinality_fn: Callable = improved_cardinality_fn,
                                       weighted_precision: bool = True, n_splits: int = 1500) -> Tuple[
        float, Dict[str, Any]]:
        """
        Parallel version of best_ts_fbeta_score using ThreadPoolExecutor.
        
        Uses threading instead of multiprocessing to avoid serialization issues
        with PyTorch tensors and instance methods.
        """
        
        # Use same parameter range as sequential version for consistency
        device = scores.device
        p_values = torch.linspace(0, 1.0, steps=n_splits, device=device)
        thresholds = torch.quantile(scores, p_values)

        label_ranges = self.compute_window_indices(labels)
        precision = torch.empty_like(thresholds, dtype=torch.float)
        recall = torch.empty_like(thresholds, dtype=torch.float)

        def process_single_threshold(idx_threshold_pair):
            """Process a single threshold computation"""
            idx, threshold = idx_threshold_pair
            
            # Create predictions for this threshold
            predictions = (scores > threshold).long()
            
            # Calculate precision and recall using instance method
            prec, rec = self.ts_precision_and_recall(
                labels,
                predictions,
                alpha=0,
                recall_cardinality_fn=recall_cardinality_fn,
                anomaly_ranges=label_ranges,
                weighted_precision=weighted_precision,
            )
            
            # Handle edge case to avoid 0/0 in F-score computation
            if prec == 0 and rec == 0:
                rec = 1
                
            return idx, prec, rec

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        # This allows us to use instance methods and share PyTorch tensors safely
        max_workers = min(16, len(thresholds))  # Don't create more threads than thresholds
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all threshold computations
            futures = {
                executor.submit(process_single_threshold, (i, t)): i
                for i, t in enumerate(thresholds)
            }

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Calculating F-beta score (parallel)"):
                idx, prec, rec = future.result()
                precision[idx] = prec
                recall[idx] = rec

        # Compute F-scores and find the best one
        f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        max_score_index = torch.argmax(f_score)

        return (
            f_score[max_score_index].item(),
            dict(
                threshold=thresholds[max_score_index].item(),
                precision=precision[max_score_index].item(),
                recall=recall[max_score_index].item(),
            ),
        )

    def __best_ts_fbeta_score_parallel_chunked(self, labels: torch.Tensor, scores: torch.Tensor, beta: float,
                                               recall_cardinality_fn: Callable = improved_cardinality_fn,
                                               weighted_precision: bool = True, n_splits: int = 1500,
                                               chunk_size: int = 10, max_workers: int = 8) -> Tuple[float, Dict[str, Any]]:
        """
        Chunked parallel version of best_ts_fbeta_score using ThreadPoolExecutor.
        
        This version processes thresholds in chunks to reduce overhead and improve efficiency.
        
        Args:
            labels: Ground truth labels
            scores: Anomaly scores  
            beta: Beta parameter for F-beta score
            recall_cardinality_fn: Cardinality function for recall calculation
            weighted_precision: Whether to use weighted precision
            n_splits: Number of threshold splits
            chunk_size: Number of thresholds to process in each chunk
            max_workers: Maximum number of worker threads
        """
        
        # Use same parameter range as sequential version for consistency
        device = scores.device
        p_values = torch.linspace(0, 1.0, steps=n_splits, device=device)
        thresholds = torch.quantile(scores, p_values)

        label_ranges = self.compute_window_indices(labels)
        precision = torch.empty_like(thresholds, dtype=torch.float)
        recall = torch.empty_like(thresholds, dtype=torch.float)
        
        def process_threshold_chunk(chunk_data):
            """Process a chunk of thresholds"""
            chunk_indices, chunk_thresholds = chunk_data
            chunk_results = []
            
            # Process each threshold in the chunk
            for i, (idx, threshold) in enumerate(zip(chunk_indices, chunk_thresholds)):
                # Create predictions for this threshold
                predictions = (scores > threshold).long()
                
                # Calculate precision and recall using instance method
                prec, rec = self.ts_precision_and_recall(
                    labels,
                    predictions,
                    alpha=0,
                    recall_cardinality_fn=recall_cardinality_fn,
                    anomaly_ranges=label_ranges,
                    weighted_precision=weighted_precision,
                )
                
                # Handle edge case to avoid 0/0 in F-score computation
                if prec == 0 and rec == 0:
                    rec = 1
                    
                chunk_results.append((idx, prec, rec))
            
            return chunk_results

        # Create chunks of threshold indices and values
        chunks = []
        for i in range(0, len(thresholds), chunk_size):
            end_idx = min(i + chunk_size, len(thresholds))
            chunk_indices = list(range(i, end_idx))
            chunk_thresholds = thresholds[i:end_idx]
            chunks.append((chunk_indices, chunk_thresholds))
        
        print(f"Processing {len(thresholds)} thresholds in {len(chunks)} chunks of size ~{chunk_size}")
        
        # Use ThreadPoolExecutor to process chunks in parallel
        actual_workers = min(max_workers, len(chunks))
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all chunk computations
            futures = {
                executor.submit(process_threshold_chunk, chunk): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Processing {len(chunks)} chunks (chunked parallel)"):
                chunk_results = future.result()
                
                # Store results in the appropriate positions
                for idx, prec, rec in chunk_results:
                    precision[idx] = prec
                    recall[idx] = rec

        # Compute F-scores and find the best one
        f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        max_score_index = torch.argmax(f_score)

        return (
            f_score[max_score_index].item(),
            dict(
                threshold=thresholds[max_score_index].item(),
                precision=precision[max_score_index].item(),
                recall=recall[max_score_index].item(),
            ),
        )

    def compute_window_indices(self, binary_labels: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Compute a list of indices where anomaly windows begin and end.

        :param binary_labels: A 1-D :class:`~torch.Tensor` containing ``1`` for an anomalous time step or ``0`` otherwise.
        :return: A list of tuples ``(start, end)`` for each anomaly window in ``binary_labels``, where ``start`` is the
            index at which the window starts and ``end`` is the first index after the end of the window.
        """
        boundaries = torch.empty_like(binary_labels)
        boundaries[0] = 0
        boundaries[1:] = binary_labels[:-1]
        boundaries *= -1
        boundaries += binary_labels
        # boundaries will be 1 where a window starts and -1 at the end of a window

        indices = torch.nonzero(boundaries, as_tuple=True)[0].tolist()
        if len(indices) % 2 != 0:
            # Add the last index as the end of a window if appropriate
            indices.append(binary_labels.shape[0])
        indices = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]

        return indices

    def _compute_overlap(self, preds: torch.Tensor, pred_indices: List[Tuple[int, int]],
                         gt_indices: List[Tuple[int, int]], alpha: float,
                         bias_fn: Callable, cardinality_fn: Callable,
                         use_window_weight: bool = False) -> float:
        n_gt_windows = len(gt_indices)
        n_pred_windows = len(pred_indices)
        total_score = 0.0
        total_gt_points = 0

        i = j = 0
        while i < n_gt_windows and j < n_pred_windows:
            gt_start, gt_end = gt_indices[i]
            window_length = gt_end - gt_start
            total_gt_points += window_length
            i += 1

            cardinality = 0
            while j < n_pred_windows and pred_indices[j][1] <= gt_start:
                j += 1
            while j < n_pred_windows and pred_indices[j][0] < gt_end:
                j += 1
                cardinality += 1

            if cardinality == 0:
                # cardinality == 0 means no overlap at all, hence no contribution
                continue

            # The last predicted window that overlaps our current window could also overlap the next window.
            # Therefore, we must consider it again in the next loop iteration.
            j -= 1

            cardinality_multiplier = cardinality_fn(cardinality, window_length)

            prediction_inside_ground_truth = preds[gt_start:gt_end]
            # We calculate omega directly in the bias function, because this can greatly improve running time
            # for the constant bias, for example.
            omega = bias_fn(prediction_inside_ground_truth)

            # Either weight evenly across all windows or based on window length
            weight = window_length if use_window_weight else 1

            # Existence reward (if cardinality > 0 then this is certainly 1)
            total_score += alpha * weight
            # Overlap reward
            total_score += (1 - alpha) * cardinality_multiplier * omega * weight

        denom = total_gt_points if use_window_weight else n_gt_windows

        return total_score / denom

    def ts_precision_and_recall(self, anomalies: torch.Tensor, predictions: torch.Tensor, alpha: float = 0,
                                recall_bias_fn: Callable[[torch.Tensor], float] = constant_bias_fn,
                                recall_cardinality_fn: Callable[[int], float] = inverse_proportional_cardinality_fn,
                                precision_bias_fn: Optional[Callable] = None,
                                precision_cardinality_fn: Optional[Callable] = None,
                                anomaly_ranges: Optional[List[Tuple[int, int]]] = None,
                                prediction_ranges: Optional[List[Tuple[int, int]]] = None,
                                weighted_precision: bool = False) -> Tuple[float, float]:
        """
        Computes precision and recall for time series as defined in [Tatbul2018]_.

        .. note::
           The default parameters for this function correspond to the defaults recommended in [Tatbul2018]_. However,
           those might not be desirable in most cases, please see [Wagner2023]_ for a detailed discussion.

        :param anomalies: Binary 1-D :class:`~torch.Tensor` of shape ``(length,)`` containing the true labels.
        :param predictions: Binary 1-D :class:`~torch.Tensor` of shape ``(length,)`` containing the predicted labels.
        :param alpha: Weight for existence term in recall.
        :param recall_bias_fn: Function that computes the bias term for a given ground-truth window.
        :param recall_cardinality_fn: Function that compute the cardinality factor for a given ground-truth window.
        :param precision_bias_fn: Function that computes the bias term for a given predicted window.
            If ``None``, this will be the same as ``recall_bias_function``.
        :param precision_cardinality_fn: Function that computes the cardinality factor for a given predicted window.
            If ``None``, this will be the same as ``recall_cardinality_function``.
        :param weighted_precision: If True, the precision score of a predicted window will be weighted with the
            length of the window in the final score. Otherwise, each window will have the same weight.
        :param anomaly_ranges: A list of tuples ``(start, end)`` for each anomaly window in ``anomalies``, where ``start``
            is the index at which the window starts and ``end`` is the first index after the end of the window. This can
            be ``None``, in which case the list is computed automatically from ``anomalies``.
        :param prediction_ranges: A list of tuples ``(start, end)`` for each anomaly window in ``predictions``, where
            ``start`` is the index at which the window starts and ``end`` is the first index after the end of the window.
            This can be ``None``, in which case the list is computed automatically from ``predictions``.
        :return: A tuple consisting of the time-series precision and recall for the given labels.
        """
        has_anomalies = torch.any(anomalies > 0).item()
        has_predictions = torch.any(predictions > 0).item()

        # Catch special cases which would cause a division by zero
        if not has_predictions and not has_anomalies:
            # In this case, the classifier is perfect, so it makes sense to set precision and recall to 1
            return 1, 1
        elif not has_predictions or not has_anomalies:
            return 0, 0

        # Set precision functions to the same as recall functions if they are not given
        if precision_bias_fn is None:
            precision_bias_fn = recall_bias_fn
        if precision_cardinality_fn is None:
            precision_cardinality_fn = recall_cardinality_fn

        if anomaly_ranges is None:
            anomaly_ranges = self.compute_window_indices(anomalies)
        if prediction_ranges is None:
            prediction_ranges = self.compute_window_indices(predictions)

        recall = self._compute_overlap(predictions, prediction_ranges, anomaly_ranges, alpha, recall_bias_fn,
                                  recall_cardinality_fn)
        precision = self._compute_overlap(anomalies, anomaly_ranges, prediction_ranges, 0, precision_bias_fn,
                                     precision_cardinality_fn, use_window_weight=weighted_precision)

        return precision, recall

    def __best_ts_fbeta_score(self, labels: torch.Tensor, scores: torch.Tensor, beta: float,
                              recall_cardinality_fn: Callable = improved_cardinality_fn,
                              weighted_precision: bool = True, n_splits: int = 1500) -> Tuple[float, Dict[str, Any]]:
        # Build thresholds from p-values (quantiles/percentiles) of the score distribution
        # p_values in [0, 1]; thresholds = percentile(scores, p_values)
        device = scores.device
        p_values = torch.linspace(0, 1.0, steps=n_splits, device=device)
        thresholds = torch.quantile(scores, p_values)
        print("Here is the shape of thresholds",thresholds.shape)
        precision = torch.empty_like(thresholds, dtype=torch.float)
        recall = torch.empty_like(thresholds, dtype=torch.float)
        predictions = torch.empty_like(scores, dtype=torch.long)

        print("Here is the shape of labels",labels.shape)
        print("Here is the shape of scores",scores.shape)
        print("Here is the shape of predictions",predictions.shape)
        print("Here is the shape of precision",precision.shape)
        print("Here is the shape of recall",recall.shape)

        label_ranges = self.compute_window_indices(labels)

        for i, t in tqdm(enumerate(thresholds), total=len(thresholds),
                        desc="Calculating F-beta score for thresholds"):
            # predictions are 0/1 longs to be compatible with downstream computations
            torch.greater(scores, t, out=predictions)
            prec, rec = self.ts_precision_and_recall(
                labels,
                predictions,
                alpha=0,
                recall_cardinality_fn=recall_cardinality_fn,
                anomaly_ranges=label_ranges,
                weighted_precision=weighted_precision,
            )

            # Avoid 0/0 in F-score computation when both prec and rec are 0
            if prec == 0 and rec == 0:
                rec = 1

            precision[i] = prec
            recall[i] = rec

        f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        max_score_index = torch.argmax(f_score)

        return (
        f_score[max_score_index].item(),
            dict(
                threshold=thresholds[max_score_index].item(),
                precision=precision[max_score_index].item(),
                recall=recall[max_score_index].item(),
            ),
        )



    def metric_PointF1PA(self, label, score, preds=None):
        import sklearn.metrics

        best_f1_adjusted = 0
        best_result = None
        q_values = np.arange(0.7, 0.99, 0.001)
        for q in tqdm(q_values, total= len(q_values), desc="Calculating PointF1PA"):
            thre = np.quantile(score, q)
            result = {}
            pred = (score > thre).astype(int)
            adjusted_pred = self.adjustment(label, pred)
            accuracy = sklearn.metrics.accuracy_score(label, adjusted_pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(label, adjusted_pred, average="binary")
            result['thre_PA'] = thre
            result['ACC_PA'] = accuracy
            result['P_PA'] = P
            result['R_PA'] = R
            result['F1_PA'] = F1
            # results.append(pd.DataFrame([result]))
            if F1 >= best_f1_adjusted:
                best_f1_adjusted = F1
                best_result = result
        if best_result is not None:
            return best_result
        else:
            assert False, "No best result found, check the input data."
        # results_storage['f1_pa'] = pd.concat(results, axis=0).reset_index(drop=True)

    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            else:
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, score, preds=None):
        from sklearn.metrics import precision_score
        true_events = self._get_events(label)

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            EventF1PA_scores = []

            for threshold in tqdm(thresholds, total=len(thresholds), desc="Calculating EventF1PA"):
                preds = (score > threshold).astype(int)

                tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
                fn = len(true_events) - tp
                rec_e = tp/(tp + fn)
                prec_t = precision_score(label, preds)
                EventF1PA = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

                EventF1PA_scores.append(EventF1PA)

            EventF1PA_Threshold = thresholds[np.argmax(EventF1PA_scores)]
            EventF1PA1 = max(EventF1PA_scores)

        else:

            tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
            fn = len(true_events) - tp
            rec_e = tp/(tp + fn)
            prec_t = precision_score(label, preds)
            EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

        return EventF1PA1

    def range_recall_new(self, labels, preds, alpha):
        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, preds)


        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))

    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair
        preds predicted data
        '''

        score = 0
        for i in labels:
            if preds[i[0]:i[1]+1].any():
                score += 1
        return score

    def num_nonzero_segments(self, x):
        count=0
        if x[0]>0:
            count+=1
        for i in range(1, len(x)):
            if x[i]>0 and x[i-1]==0:
                count+=1
        return count

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]


            x1 = np.arange(e,min(e+window//2,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))

            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))

        label = np.minimum(np.ones(length), label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            l0 = int((e-s+1)*percentage)

            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))

            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))

        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        indices = np.where(labels == 1)[0]
        product = labels * pred
        TP = np.sum(product)
        newlabels = product.copy()
        newlabels[indices] = 1

        # recall = min(TP/P,1)
        P_new = (P + np.sum(newlabels)) / 2  # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP / P_new, 1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))

        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:  # if newlabels>0, that segment must contained
                existence += 1

        existence_ratio = existence / len(L)
        # print(existence_ratio)

        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall * existence_ratio

        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))

        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP / N_new

        Precision_RangeAUC = TP / np.sum(pred)

        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)

        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type == 'window':
            labels = self.extend_postive_range(labels, window=window)
        else:
            labels = self.extend_postive_range_individual(labels, percentage=percentage)

        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]

        for i in np.linspace(0, len(score) - 1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score >= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P, L)

            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)

        TPR_list.append(1)
        FPR_list.append(1)  # otherwise, range-AUC will stop earlier than (1,1)

        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)

        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        AUC_range = np.sum(width * height)

        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = prec[1:]
        AP_range = np.sum(width_PR * height_PR)

        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec

        return AUC_range

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))

    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    # TPR_FPR_window
    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
                    if (p[j][seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                N_labels = 0
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio

                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = (AP_range)
        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)


    def metric_VUS_pred(self, labels, preds, windowSize):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels)
        seq = self.range_convers_new(labels)
        l = self.new_sequence(labels, seq, windowSize)

        recall_3d = np.zeros((windowSize + 1))
        prec_3d = np.zeros((windowSize + 1))
        f_3d = np.zeros((windowSize + 1))

        N_pred = np.sum(preds)

        for window in window_3d:

            labels_extended = self.sequencing(labels, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            labels = labels_extended.copy()
            existence = 0

            for seg in L:
                labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * preds[seg[0]:seg[1] + 1]
                if (preds[seg[0]:(seg[1] + 1)] > 0).any():
                    existence += 1
            for seg in seq:
                labels[seg[0]:seg[1] + 1] = 1

            TP = 0
            N_labels = 0
            for seg in l:
                TP += np.dot(labels[seg[0]:seg[1] + 1], preds[seg[0]:seg[1] + 1])
                N_labels += np.sum(labels[seg[0]:seg[1] + 1])

            P_new = (P + N_labels) / 2
            recall = min(TP / P_new, 1)
            Precision = TP / N_pred

            recall_3d[window] = recall
            prec_3d[window] = Precision
            f_3d[window] = 2 * Precision * recall / (Precision + recall) if (Precision + recall) > 0 else 0
        return sum(recall_3d) / len(window_3d), sum(prec_3d) / len(window_3d), sum(f_3d) / len(window_3d)

    def metric_Affiliation_optimized(self, label, score, num_workers=None):
        """
        Optimized version with ThreadPool and better chunking
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)

        print(f"Computing Affiliation (optimized) with {num_workers} workers")
        start_time = time.time()

        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        # Pre-compute ground truth events once
        events_gt = convert_vector_to_events(label)
        Trange = (0, len(label))

        # Generate p-values and thresholds
        p_values = np.linspace(0.8, 1, 300)

        # Pre-compute all thresholds
        thresholds = np.quantile(score, p_values)

        # Pre-compute all predictions
        print("Pre-computing predictions...")
        all_predictions = []
        for threshold in thresholds:
            preds = (score > threshold).astype(int)
            all_predictions.append(preds)

        print("Computing affiliation metrics...")

        # Function to process a batch of indices
        def compute_metrics_batch(indices):
            results = []
            for idx in indices:
                preds = all_predictions[idx]

                events_pred = convert_vector_to_events(preds)
                affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

                prec = affiliation_metrics['Affiliation_Precision']
                rec = affiliation_metrics['Affiliation_Recall']

                if prec + rec > 0:
                    f1 = 2 * prec * rec / (prec + rec + self.eps)
                else:
                    f1 = 0.0

                results.append({
                    'f1': f1,
                    'precision': prec,
                    'recall': rec,
                    'p_value': p_values[idx],
                    'threshold': thresholds[idx]
                })

            return results

        # Split indices for workers
        indices = list(range(len(p_values)))
        chunk_size = len(indices) // num_workers
        if chunk_size == 0:
            chunk_size = 1
        index_chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

        # Process with thread pool
        all_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(compute_metrics_batch, chunk) for chunk in index_chunks]

            completed = 0
            for future in as_completed(futures):
                all_results.extend(future.result())
                completed += 1
                print(f"Progress: {completed}/{len(futures)} chunks completed", end='\r')

        print()  # New line

        # Find best result
        best_result = max(all_results, key=lambda x: x['f1'])

        elapsed = time.time() - start_time
        print(f"Affiliation computed in {elapsed:.2f}s")

        return best_result['f1'], best_result['precision'], best_result['recall']

    def metric_Affiliation_chunked(self, label, score, chunk_size=30, num_workers=4):
        """
        Simple chunked parallel processing
        """
        print(f"Computing Affiliation (chunked) with {num_workers} workers, chunk_size={chunk_size}")
        start_time = time.time()

        # Generate p-values
        p_values = np.linspace(0.8, 1, 300)

        # Create chunks of p-values
        p_value_chunks = [p_values[i:i + chunk_size]
                          for i in range(0, len(p_values), chunk_size)]

        # Prepare arguments for workers
        chunk_args = [(chunk, label, score) for chunk in p_value_chunks]

        # Process in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, result_chunk in enumerate(executor.map(self._process_affiliation_chunk, chunk_args)):
                all_results.extend(result_chunk)
                print(f"Progress: {(i + 1) * chunk_size}/{len(p_values)} thresholds processed", end='\r')

        print()  # New line

        # Find best result
        best_result = max(all_results, key=lambda x: x['f1'])

        elapsed = time.time() - start_time
        print(f"Affiliation computed in {elapsed:.2f}s")

        return best_result['f1'], best_result['precision'], best_result['recall']

    def _compute_affiliation_chunk(self, p_values_chunk, score, label, eps=1e-7):
        """
        Process a chunk of p-values for affiliation metrics
        """
        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events
        
        # Ensure proper data types to avoid float/integer issues
        label = np.asarray(label, dtype=int)
        score = np.asarray(score, dtype=float)
        
        # Convert ground truth to events once for this chunk
        events_gt = convert_vector_to_events(label)
        Trange = (0, len(label))
        
        chunk_results = []
        for p in p_values_chunk:
            threshold = np.quantile(score, p)
            preds_loop = (score > threshold).astype(int)
            
            events_pred = convert_vector_to_events(preds_loop)
            affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
            
            Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
            Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
            
            denominator = Affiliation_Precision + Affiliation_Recall
            if denominator > 0:
                Affiliation_F = 2 * Affiliation_Precision * Affiliation_Recall / (denominator + eps)
            else:
                Affiliation_F = 0.0
            
            chunk_results.append({
                'f1': Affiliation_F,
                'precision': Affiliation_Precision,
                'recall': Affiliation_Recall,
                'p_value': p,
                'threshold': threshold
            })
        
        return chunk_results

    def _compute_affiliation_parallel(self, label, score, num_workers=8):
        """
        Parallel computation with progress bar
        """
        print(f"Computing Affiliation (parallel) with {num_workers} workers")
        start_time = time.time()

        # Generate p-values
        p_values = np.linspace(0.8, 1, 300)
        total_thresholds = len(p_values)

        # Split p-values into chunks for parallel processing
        p_value_chunks = np.array_split(p_values, num_workers)

        # Process chunks in parallel with progress bar
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks and track chunk sizes
            futures = {}
            for i, chunk in enumerate(p_value_chunks):
                future = executor.submit(self._compute_affiliation_chunk, chunk, score, label)
                futures[future] = len(chunk)

            # Collect results with progress bar
            all_results = []
            with tqdm(
                    total=total_thresholds,
                    desc="Computing affiliation metrics",
                    unit="threshold",
                    colour="green"
            ) as pbar:
                for future in as_completed(futures):
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    # Update by the number of thresholds processed in this chunk
                    pbar.update(futures[future])

        # Find best result
        best_result = max(all_results, key=lambda x: x['f1'])

        elapsed = time.time() - start_time
        print(f"Affiliation computed in {elapsed:.2f}s")

        return best_result['f1'], best_result['precision'], best_result['recall']

    def metric_Affiliation_optimized(self, label, score, num_workers=None):
        """
        Optimized version with ThreadPool and better chunking
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)

        print(f"Computing Affiliation (optimized) with {num_workers} workers")
        start_time = time.time()

        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        # Pre-compute ground truth events once
        events_gt = convert_vector_to_events(label)
        Trange = (0, len(label))

        # Generate p-values and thresholds
        p_values = np.linspace(0.8, 1, 300)

        # Pre-compute all thresholds
        thresholds = np.quantile(score, p_values)

        # Pre-compute all predictions
        print("Pre-computing predictions...")
        all_predictions = []
        for threshold in thresholds:
            preds = (score > threshold).astype(int)
            all_predictions.append(preds)

        print("Computing affiliation metrics...")

        # Function to process a batch of indices
        def compute_metrics_batch(indices):
            results = []
            for idx in indices:
                preds = all_predictions[idx]

                events_pred = convert_vector_to_events(preds)
                affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

                prec = affiliation_metrics['Affiliation_Precision']
                rec = affiliation_metrics['Affiliation_Recall']

                if prec + rec > 0:
                    f1 = 2 * prec * rec / (prec + rec + self.eps)
                else:
                    f1 = 0.0

                results.append({
                    'f1': f1,
                    'precision': prec,
                    'recall': rec,
                    'p_value': p_values[idx],
                    'threshold': thresholds[idx]
                })

            return results

        # Split indices for workers
        indices = list(range(len(p_values)))
        chunk_size = len(indices) // num_workers
        if chunk_size == 0:
            chunk_size = 1
        index_chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

        # Process with thread pool
        all_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(compute_metrics_batch, chunk) for chunk in index_chunks]

            completed = 0
            for future in as_completed(futures):
                all_results.extend(future.result())
                completed += 1
                print(f"Progress: {completed}/{len(futures)} chunks completed", end='\r')

        print()  # New line

        # Find best result
        best_result = max(all_results, key=lambda x: x['f1'])

        elapsed = time.time() - start_time
        print(f"Affiliation computed in {elapsed:.2f}s")

        return best_result['f1'], best_result['precision'], best_result['recall']

    def metric_Affiliation_chunked(self, label, score, chunk_size=30, num_workers=4):
        """
        Simple chunked parallel processing
        """
        print(f"Computing Affiliation (chunked) with {num_workers} workers, chunk_size={chunk_size}")
        start_time = time.time()

        # Generate p-values
        p_values = np.linspace(0.8, 1, 300)

        # Create chunks of p-values
        p_value_chunks = [p_values[i:i + chunk_size]
                          for i in range(0, len(p_values), chunk_size)]

        # Prepare arguments for workers
        chunk_args = [(chunk, label, score) for chunk in p_value_chunks]

        # Process in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, result_chunk in enumerate(executor.map(self._process_affiliation_chunk, chunk_args)):
                all_results.extend(result_chunk)
                print(f"Progress: {(i + 1) * chunk_size}/{len(p_values)} thresholds processed", end='\r')

        print()  # New line

        # Find best result
        best_result = max(all_results, key=lambda x: x['f1'])

        elapsed = time.time() - start_time
        print(f"Affiliation computed in {elapsed:.2f}s")

        return best_result['f1'], best_result['precision'], best_result['recall']

    @staticmethod
    def _process_affiliation_chunk(args):
        """
        Static method to process a chunk of p-values for affiliation metrics.
        This can be pickled for multiprocessing.
        """
        chunk_p_values, label_local, score_local = args
        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        # Convert ground truth to events once for this chunk
        events_gt = convert_vector_to_events(label_local)
        Trange = (0, len(label_local))

        results = []
        for p in chunk_p_values:
            threshold = np.quantile(score_local, p)
            preds = (score_local > threshold).astype(int)

            events_pred = convert_vector_to_events(preds)
            affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

            prec = affiliation_metrics['Affiliation_Precision']
            rec = affiliation_metrics['Affiliation_Recall']

            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec + 1e-7)
            else:
                f1 = 0.0

            results.append({
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'p_value': p,
                'threshold': threshold
            })

        return results

    def metric_Affiliation_sampling(self, label, score, sample_rate=0.2):
        """
        Fast approximation by sampling thresholds
        """
        print(f"Computing Affiliation with threshold sampling (rate={sample_rate})")
        start_time = time.time()

        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        # Convert ground truth to events once
        events_gt = convert_vector_to_events(label)
        Trange = (0, len(label))

        # Generate fewer p-values
        n_samples = int(300 * sample_rate)
        p_values = np.linspace(0.8, 1, n_samples)

        results = []
        for p in tqdm(p_values, desc="Sampling affiliation", unit="threshold"):
            threshold = np.quantile(score, p)
            preds = (score > threshold).astype(int)

            events_pred = convert_vector_to_events(preds)
            affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

            prec = affiliation_metrics['Affiliation_Precision']
            rec = affiliation_metrics['Affiliation_Recall']

            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec + self.eps)
            else:
                f1 = 0.0

            results.append({
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'p_value': p,
                'threshold': threshold
            })

        # Find best result
        best_result = max(results, key=lambda x: x['f1'])

        elapsed = time.time() - start_time
        print(f"Affiliation computed in {elapsed:.2f}s (approximate)")

        return best_result['f1'], best_result['precision'], best_result['recall']

    def metric_standard_F1_chunked(self, true_labels, anomaly_scores, threshold=None, chunk_size=50, num_workers=4):
        """
        Optimized chunked parallel version of metric_standard_F1.
        
        Calculate F1, Precision, Recall using parallel threshold processing.

        Args:
            true_labels: np.ndarray, ground truth binary labels (0=normal, 1=anomaly)
            anomaly_scores: np.ndarray, anomaly scores (continuous values)
            threshold: float, optional. If None, will use optimal threshold based on F1 score
            chunk_size: int, number of thresholds to process in each chunk
            num_workers: int, number of parallel workers

        Returns:
            dict: Dictionary containing various metrics
        """
        # If threshold is provided, use original method
        if threshold is not None:
            return self.metric_standard_F1(true_labels, anomaly_scores, threshold)
            
        print(f"Computing standard F1 (chunked) with {num_workers} workers, chunk_size={chunk_size}")
        start_time = time.time()

        # Generate thresholds
        thresholds = np.linspace(0.5, 1, 500)
        total_thresholds = len(thresholds)

        # Create chunks of thresholds
        threshold_chunks = [thresholds[i:i + chunk_size]
                           for i in range(0, len(thresholds), chunk_size)]

        print(f"Split {total_thresholds} thresholds into {len(threshold_chunks)} chunks")

        # Process in parallel
        all_results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=total_thresholds, desc="Processing standard F1 thresholds", unit="threshold", colour="blue") as pbar:
                # Prepare arguments
                chunk_args = [(chunk, true_labels, anomaly_scores) for chunk in threshold_chunks]

                # Process and update progress bar
                for i, result_chunk in enumerate(executor.map(self._process_standard_f1_chunk, chunk_args)):
                    all_results.extend(result_chunk)
                    pbar.update(len(threshold_chunks[i]))
                    pbar.set_postfix({
                        'chunk': f"{i + 1}/{len(threshold_chunks)}",
                        'results': len(all_results)
                    })

        # Find best result
        best_result = max(all_results, key=lambda x: x['f1'])

        elapsed = time.time() - start_time
        print(f"✓ Standard F1 computed in {elapsed:.2f}s")
        print(f"  Best F1: {best_result['f1']:.4f} at threshold {best_result['threshold']:.4f}")

        return {
            'F1': best_result['f1'],
            'Recall': best_result['recall'],
            'Precision': best_result['precision']
        }

    @staticmethod
    def _process_standard_f1_chunk(args):
        """
        Static method to process a chunk of thresholds for standard F1 metrics.
        This can be pickled for multiprocessing.
        """
        chunk_thresholds, true_labels, anomaly_scores = args
        results = []

        for t in chunk_thresholds:
            threshold = np.quantile(anomaly_scores, t)
            predictions = (anomaly_scores >= threshold).astype(int)
            
            if len(np.unique(predictions)) > 1:  # Avoid division by zero
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predictions, average='binary', zero_division=0
                )
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0

            results.append({
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'threshold': threshold,
                'quantile': t
            })

        return results

    def metric_PointF1PA_chunked(self, label, score, preds=None, chunk_size=50, num_workers=4):
        """
        Optimized chunked parallel version of metric_PointF1PA.
        
        Calculate Point F1 with Point Adjustment using parallel threshold processing.

        Args:
            label: np.ndarray, ground truth binary labels
            score: np.ndarray, anomaly scores
            preds: np.ndarray, optional. If provided, use these predictions directly
            chunk_size: int, number of thresholds to process in each chunk
            num_workers: int, number of parallel workers

        Returns:
            dict: Dictionary containing various metrics (same format as original method)
        """
        # If predictions are provided, use original method
        if preds is not None:
            return self.metric_PointF1PA(label, score, preds)
            
        print(f"Computing PointF1PA (chunked) with {num_workers} workers, chunk_size={chunk_size}")
        start_time = time.time()

        # Generate q_values (quantiles)
        q_values = np.arange(0.7, 0.99, 0.001)
        total_thresholds = len(q_values)

        # Create chunks of q_values
        q_value_chunks = [q_values[i:i + chunk_size]
                         for i in range(0, len(q_values), chunk_size)]

        print(f"Split {total_thresholds} thresholds into {len(q_value_chunks)} chunks")

        # Process in parallel
        all_results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=total_thresholds, desc="Processing PointF1PA thresholds", unit="threshold", colour="green") as pbar:
                # Prepare arguments
                chunk_args = [(chunk, label, score) for chunk in q_value_chunks]

                # Process and update progress bar
                for i, result_chunk in enumerate(executor.map(self._process_pointf1pa_chunk, chunk_args)):
                    all_results.extend(result_chunk)
                    pbar.update(len(q_value_chunks[i]))
                    pbar.set_postfix({
                        'chunk': f"{i + 1}/{len(q_value_chunks)}",
                        'results': len(all_results)
                    })

        # Find best result
        best_result = max(all_results, key=lambda x: x['F1_PA'])

        elapsed = time.time() - start_time
        print(f"✓ PointF1PA computed in {elapsed:.2f}s")
        print(f"  Best F1_PA: {best_result['F1_PA']:.4f} at threshold {best_result['thre_PA']:.4f}")

        return best_result

    @staticmethod
    def _process_pointf1pa_chunk(args):
        """
        Static method to process a chunk of q_values for PointF1PA metrics.
        This can be pickled for multiprocessing.
        """
        import sklearn.metrics
        
        chunk_q_values, label, score = args
        results = []

        # Create a basic_metricor instance to access adjustment method
        grader = basic_metricor()

        for q in chunk_q_values:
            thre = np.quantile(score, q)
            pred = (score > thre).astype(int)
            adjusted_pred = grader.adjustment(label, pred)
            
            accuracy = sklearn.metrics.accuracy_score(label, adjusted_pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(label, adjusted_pred, average="binary")
            
            result = {
                'thre_PA': thre,
                'ACC_PA': accuracy,
                'P_PA': P,
                'R_PA': R,
                'F1_PA': F1,
                'quantile': q
            }
            
            results.append(result)

        return results



class SPOT:
    def __init__(self, q=1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, verbose=True):
        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]

        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        zeros = np.concatenate((left_zeros, right_zeros))

        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True):
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        th = []
        alarm = []
        for i in tqdm(range(self.data.size)):

            if self.data[i] > self.extreme_quantile:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)

            elif self.data[i] > self.init_threshold:
                self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                self.Nt += 1
                self.n += 1

                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)
            else:
                self.n += 1

            th.append(self.extreme_quantile)

        return {'thresholds': th, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig