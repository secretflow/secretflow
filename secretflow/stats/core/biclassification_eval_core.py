# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a single party based bi-classification report

from typing import List, Tuple, Union

import jax

import jax.numpy as jnp
import pandas as pd

from .utils import equal_obs, equal_range


class Report:
    """Report containing all other reports for bi-classification evaluation

    Attributes:
        summary_report: SummaryReport

        eq_frequent_bin_report: List[EqBinReport]

        eq_range_bin_report: List[EqBinReport]

        head_report: List[PrReport]
            reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2
    """

    def __init__(
        self,
        eq_frequent_result_arr_list,
        eq_range_result_arr_list,
        summary_report_arr,
        head_prs,
    ):
        self.eq_frequent_bin_report = [
            EqBinReport(a) for a in eq_frequent_result_arr_list
        ]
        self.eq_range_bin_report = [EqBinReport(a) for a in eq_range_result_arr_list]
        self.summary_report = SummaryReport(summary_report_arr)
        self.head_report = [PrReport(a) for a in head_prs]


class PrReport:
    """Precision Related statistics Report.

    Attributes:
        fpr: float
            FP/(FP+TN)
        precision: float
            TP/(TP+FP)
        recall: float
            TP/(TP+FN)
    """

    def __init__(self, arr):
        assert arr.size == PR_REPORT_STATISTICS_ENTRY_COUNT
        self.fpr = arr[0]
        self.precision = arr[1]
        self.recall = arr[2]
        self.threshold = arr[3]


PR_REPORT_STATISTICS_ENTRY_COUNT = 4


class SummaryReport:
    """Summary Report for bi-classification evaluation.

    Attributes:
        total_samples: int

        positive_samples: int

        negative_samples: int

        auc: float
            auc: area under the curve: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
        ks: float
            Kolmogorov-Smirnov statistics: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
        f1_score: float
            harmonic mean of precision and recall: https://en.wikipedia.org/wiki/F-score
    """

    def __init__(self, arr):
        assert arr.size == SUMMARY_REPORT_STATISTICS_ENTRY_COUNT
        self.total_samples = arr[0]
        self.positive_samples = arr[1]
        self.negative_samples = arr[2]
        self.auc = arr[3]
        self.ks = arr[4]
        self.f1_score = arr[5]


SUMMARY_REPORT_STATISTICS_ENTRY_COUNT = 6


class GroupReport:
    """Report for each group"""

    group_name: str
    summary: SummaryReport


class EqBinReport:
    """Statistics Report for each bin.

    Attributes:

        start_value: float

        end_value: float

        positive: int

        negative: int

        total: int

        precision: float

        recall: float

        false_positive_rate: float

        f1_score: float

        lift: float
            see https://en.wikipedia.org/wiki/Lift_(data_mining)
        predicted_positive_ratio: float
            predicted positive samples / total samples.
        predicted_negative_ratio: float
            predicted negative samples / total samples.
        cumulative_percent_of_positive: float

        cumulative_percent_of_negative: float

        total_cumulative_percent: float

        ks: float

        avg_score: float
    """

    def __init__(self, arr):
        # assert arr.size == BIN_REPORT_STATISTICS_ENTRY_COUNT, "{}, {}".format(arr.size, BIN_REPORT_STATISTICS_ENTRY_COUNT)
        self.start_value = arr[0]
        self.end_value = arr[1]
        self.positive = arr[2]
        self.negative = arr[3]
        self.total = arr[4]
        self.precision = arr[5]
        self.recall = arr[6]
        self.false_positive_rate = arr[7]
        self.f1_score = arr[8]
        self.Lift = arr[9]
        self.predicted_positive_ratio = arr[10]
        self.predicted_negative_ratio = arr[11]
        self.cumulative_percent_of_positive = arr[12]
        self.cumulative_percent_of_negative = arr[13]
        self.total_cumulative_percent = arr[14]
        self.ks = arr[15]
        self.avg_score = arr[16]


HEAD_FPR_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

BIN_REPORT_STATISTICS_ENTRY_COUNT = 17


def gen_all_reports(
    y_true: Union[pd.DataFrame, jnp.array],
    y_score: Union[pd.DataFrame, jnp.array],
    bin_size: int,
    min_item_cnt_per_bucket: int = None,
):
    """Generate all reports.

    Args:
        y_true: Union[pd.DataFrame, jnp.array]
            should be of shape n * 1 and with binary entries
            1 means positive sample
        y_score: Union[pd.DataFrame, jnp.array]
            should be of shape n * 1 and with each entry between [0, 1]
            probability of being positive
        bin_size: int
            number of bins to evaluate
        min_item_cnt_per_bucket: int
            min item cnt per bucket. If any bucket doesn't meet the requirement, return NaN values.
    Returns:

    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    if isinstance(y_score, pd.DataFrame):
        y_score = y_score.to_numpy()
    sorted_label_score_pair_arr = create_sorted_label_score_pair(y_true, y_score)
    pos_count = jnp.sum(y_true)
    eq_frequent_result_arr_list = eq_frequent_bin_evaluate(
        sorted_label_score_pair_arr, pos_count, bin_size, min_item_cnt_per_bucket
    )
    eq_range_result_arr_list = eq_range_bin_evaluate(
        sorted_label_score_pair_arr, pos_count, bin_size, min_item_cnt_per_bucket
    )
    # fill summary report
    # positive has index 2
    positive_samples = jnp.sum(
        jnp.array([bin[2] for bin in eq_frequent_result_arr_list])
    )
    # negative has index 3
    negative_samples = jnp.sum(
        jnp.array([bin[3] for bin in eq_frequent_result_arr_list])
    )
    # ks has index 15
    ks = jnp.max(jnp.array([bin[15] for bin in eq_frequent_result_arr_list]))
    # f1 has index 8
    f1 = jnp.max(jnp.array([bin[8] for bin in eq_frequent_result_arr_list]))

    total_samples = positive_samples + negative_samples
    auc = binary_roc_auc(sorted_label_score_pair_arr)
    summary_report_arr = jnp.array(
        [total_samples, positive_samples, negative_samples, auc, ks, f1]
    )

    # fill head prs
    head_prs = gen_pr_reports(
        sorted_label_score_pair_arr, jnp.array(HEAD_FPR_THRESHOLDS)
    )
    return Report(
        eq_frequent_result_arr_list,
        eq_range_result_arr_list,
        summary_report_arr,
        head_prs,
    )


def create_sorted_label_score_pair(y_true: jnp.array, y_score: jnp.array):
    """produce an n * 2 shaped array with the second column as the sorted scores, in decreasing order"""
    unsorted_array = jnp.concatenate([y_true, y_score], axis=1)
    return unsorted_array[jnp.argsort(unsorted_array[:, 1])[::-1]]


def eq_frequent_bin_evaluate(
    sorted_pairs: jnp.array,
    pos_count: int,
    bin_size: int,
    min_item_cnt_per_bucket: int = None,
) -> List[jnp.array]:
    """Fill eq frequent bin report.

    Args:
        sorted_pairs: jnp.array
            Should be of shape n * 2 and with second col sorted
        pos_count: int
            Total number of positive samples
        bin_size: int
            Total number of bins
        min_item_cnt_per_bucket: int
            min item cnt per bucket. If any bucket doesn't meet the requirement, error raises.
    Returns:
        bin_reports: List[jnp.array]

    """
    # split points have length bin_size + 1
    split_points = equal_obs(sorted_pairs[:, 1], bin_size)
    # split points should be reversed to become a decreasing sequence
    split_points = jnp.flip(split_points)

    # Each bin has domain (split_left, split_right]
    return evaluate_bins(sorted_pairs, pos_count, split_points, min_item_cnt_per_bucket)


def eq_range_bin_evaluate(
    sorted_pairs: jnp.array,
    pos_count: int,
    bin_size: int,
    min_item_cnt_per_bucket: int = None,
) -> List[jnp.array]:
    """Fill eq range bin report.

    Args:
        sorted_pairs: jnp.array
            Should be of shape n * 2 and with second col sorted.
        pos_count: int
            Total number of positive samples
        bin_size: int
            Total number of bins
        min_item_cnt_per_bucket: int
            min item cnt per bucket. If any bucket doesn't meet the requirement, will return nan values.
    Returns:
        bin_reports: List[jnp.array]

    """
    # split points have length bin_size + 1
    split_points = equal_range(sorted_pairs[:, 1], bin_size)
    # split points should be reversed to become a decreasing sequence
    split_points = jnp.flip(split_points)

    # Each bin has domain (split_left, split_right]
    return evaluate_bins(sorted_pairs, pos_count, split_points, min_item_cnt_per_bucket)


@jax.jit
def get_end_positions(x, split_points):
    end_positions = jnp.sum(x[:, None] > split_points, axis=0)
    return end_positions


def evaluate_bins(
    sorted_pairs: jnp.array, pos_count: int, split_points, min_item_cnt_per_bucket
) -> List[jnp.array]:
    """evaluate bins given sorted pairs, pos_count and split_points (in decreasing order)"""
    n_samples = sorted_pairs.shape[0]
    neg_count = n_samples - pos_count
    cumulative_pos_count = 0
    cumulative_neg_count = 0
    start_pos = 0
    end_pos = 0
    bins = []
    end_pos_new = get_end_positions(sorted_pairs[:, 1], split_points)
    for end_pos in end_pos_new:
        # problematic case
        if (
            (min_item_cnt_per_bucket is not None)
            and (end_pos - start_pos) < min_item_cnt_per_bucket
            and (end_pos - start_pos) > 0
        ):
            # append enpty bin_report_arr
            t = bin_evaluate(
                sorted_pairs,
                start_pos,
                end_pos,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
            )
            bin_report_arr, cumulative_pos_count, cumulative_neg_count = (
                t[0],
                t[1],
                t[2],
            )
        else:
            t = bin_evaluate(
                sorted_pairs,
                start_pos,
                end_pos,
                pos_count,
                neg_count,
                cumulative_pos_count,
                cumulative_neg_count,
            )
            bin_report_arr, cumulative_pos_count, cumulative_neg_count = (
                t[0],
                t[1],
                t[2],
            )
        bins.append(bin_report_arr)
        start_pos = end_pos

    # last bin
    bin_report_arr, _, _ = bin_evaluate(
        sorted_pairs,
        start_pos,
        n_samples,
        pos_count,
        neg_count,
        cumulative_pos_count,
        cumulative_neg_count,
    )
    bins.append(bin_report_arr)
    return bins


def bin_evaluate(
    sorted_pairs,
    start_pos,
    end_pos,
    total_pos_count,
    total_neg_count,
    cumulative_pos_count,
    cumulative_neg_count,
) -> Tuple[jnp.array, int, int]:
    """Evaluate statistics for a bin.

    Returns:
        bin_report_arr: jnp.array
            an array of size BIN_REPORT_STATISTICS_ENTRY_COUNT

        cumulative_pos_count: int

        cumulative_neg_count: int

    """
    if end_pos == start_pos:
        return jnp.zeros((BIN_REPORT_STATISTICS_ENTRY_COUNT, 1))

    # compute new f1
    (
        true_positive,
        true_negative,
        false_positive,
        false_negative,
    ) = confusion_matrix_from_cum_counts(
        cumulative_pos_count, cumulative_neg_count, total_neg_count, total_pos_count
    )

    pos_count = jnp.sum(sorted_pairs[start_pos:end_pos, 0])
    neg_count = end_pos - start_pos - pos_count
    score_sum = jnp.sum(sorted_pairs[start_pos:end_pos, 1])
    false_negative -= pos_count
    true_positive += pos_count
    true_negative -= neg_count
    false_positive += neg_count
    f1_score = compute_f1_score(true_positive, false_positive, false_negative)

    # fill in rest of eq_bin_reports
    start_value = float(sorted_pairs[end_pos - 1, 1])
    end_value = float(sorted_pairs[start_pos, 1])
    positive = int(pos_count)
    negative = int(neg_count)
    total = int(end_pos - start_pos)
    precision, recall, false_positive_rate = precision_recall_false_positive_rate(
        true_positive, false_positive, false_negative, true_negative
    )

    f1_score = float(f1_score)
    lift = float(precision * (total_pos_count + total_neg_count) / total_pos_count)
    predicted_positive_ratio = float(pos_count / total_pos_count)
    predicted_negative_ratio = float(neg_count / total_neg_count)
    cumulative_percent_of_positive = float(
        (pos_count + cumulative_pos_count) / total_pos_count
    )
    cumulative_percent_of_negative = float(
        (neg_count + cumulative_neg_count) / total_neg_count
    )
    total_cumulative_percent = float(
        (pos_count + cumulative_pos_count + neg_count + cumulative_neg_count)
        / (total_pos_count + total_neg_count)
    )
    ks = abs(float(cumulative_percent_of_positive - cumulative_percent_of_negative))

    avg_score = float(score_sum / total)
    # pack into a single array
    bin_report_arr = jnp.array(
        [
            start_value,
            end_value,
            positive,
            negative,
            total,
            precision,
            recall,
            false_positive_rate,
            f1_score,
            lift,
            predicted_positive_ratio,
            predicted_negative_ratio,
            cumulative_percent_of_positive,
            cumulative_percent_of_negative,
            total_cumulative_percent,
            ks,
            avg_score,
        ]
    )
    assert bin_report_arr.size == BIN_REPORT_STATISTICS_ENTRY_COUNT, "{}, {}".format(
        bin_report_arr.size, BIN_REPORT_STATISTICS_ENTRY_COUNT
    )

    # update cumulative values
    cumulative_pos_count += pos_count
    cumulative_neg_count += neg_count
    return bin_report_arr, cumulative_pos_count, cumulative_neg_count


def gen_pr_reports(sorted_pairs: jnp.array, thresholds: jnp.array) -> List[jnp.array]:
    """Generate pr report per specified threshold.

    Args:
        sorted_pairs: jnp.array
            y_true y_score pairs sorted by y_score in increasing order
            shape n_samples * 2
        thresholds: 1d jnp.ndarray
            prediction thresholds on which to evaluate
    Returns:
       pr_report_arr: List[jnp.array]
        a list of pr reports in jnp.array of shape 3 * 1, list len = len(thresholds)
    """
    tps, fps, all_thresholds = binary_clf_curve(sorted_pairs)
    n_positive = tps[-1]
    n_negative = fps[-1]

    result = []
    for t in thresholds:
        i = jnp.sum(all_thresholds < t)
        precision = tps[i] / (tps[i] + fps[i])
        recall = tps[i] / n_positive
        false_positive_rate = fps[i] / n_negative
        pr_report = jnp.array([false_positive_rate, precision, recall, t])
        result.append(pr_report)
    return result


# section of statistics
def precision_recall_false_positive_rate(
    true_positive, false_positive, false_negative, true_negative
) -> Tuple[float, float, float]:
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (false_positive + true_negative)
    return float(precision), float(recall), float(false_positive_rate)


def confusion_matrix_from_cum_counts(
    cumulative_pos_count, cumulative_neg_count, total_neg_count, total_pos_count
):
    """Compute the confusion matrix.

    Args:
        cumulative_pos_count: int

        cumulative_neg_count: int

        total_neg_count: int

        total_pos_count: int

    Returns:
        true_positive: int

        true_negative: int

        false_positive: int

        false_negative: int

    """
    true_positive = cumulative_pos_count
    true_negative = total_neg_count - cumulative_neg_count
    false_positive = cumulative_neg_count
    false_negative = total_pos_count - cumulative_pos_count
    return true_positive, true_negative, false_positive, false_negative


def binary_clf_curve(sorted_pairs: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve).

    Args:
        sorted_pairs: jnp.array
            y_true y_score pairs sorted by y_score in decreasing order
    Returns:
        fps: 1d ndarray
            False positives counts, index i records the number
            of negative samples that got assigned a
            score >= thresholds[i].
            The total number of negative samples is equal to
            fps[-1] (thus true negatives are given by fps[-1] - fps)
        tps: 1d ndarray
            True positives counts, index i records the number
            of positive samples that got assigned a
            score >= thresholds[i].
            The total number of positive samples is equal to
            tps[-1] (thus false negatives are given by tps[-1] - tps)
        thresholds : 1d ndarray
            Distinct predicted score sorted in decreasing order
    References:
        Github: scikit-learn _binary_clf_curve.
    """
    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve

    distinct_indices = jnp.where(jnp.diff(sorted_pairs[:, 1]))[0]
    end = jnp.array([sorted_pairs.shape[0] - 1])
    threshold_indices = jnp.hstack((distinct_indices, end))

    thresholds = sorted_pairs[threshold_indices, 1]
    tps = jnp.cumsum(sorted_pairs[:, 0])[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return fps, tps, thresholds


def roc_curve(sorted_pairs: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """Compute Receiver operating characteristic (ROC).

    Compared to sklearn implementation, this implementation eliminates most conditionals and ill-conditionals checking.

    Args:
        sorted_pairs: jnp.array
            y_true y_score pairs sorted by y_score in decreasing order
    Returns:
        fpr: ndarray of shape (>2,)
            Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= `thresholds[i]`.
        tpr: ndarray of shape (>2,)
            Increasing true positive rates such that element `i` is the true
            positive rate of predictions with score >= `thresholds[i]`.
        thresholds: ndarray of shape = (n_thresholds,)
            Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
    References:
        Github: scikit-learn roc_curve.
    """
    fps, tps, thresholds = binary_clf_curve(sorted_pairs)
    tps = jnp.r_[0, tps]
    fps = jnp.r_[0, fps]
    thresholds = jnp.r_[thresholds[0] + 1, thresholds]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return fpr, tpr, thresholds


@jax.jit
def auc(x, y):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.
    X must be monotonic, no checking inside function.

    Args:
        x: ndarray of shape (n,)
            monotonic X coordinates
        y: ndarray of shape, (n,)
            Y coordinates
    Returns:
        auc: float
            Area Under the Curve
    """
    x, y = jax.lax.sort([x, y], num_keys=1)
    area = jnp.abs(jnp.trapz(y, x))
    return area


def binary_roc_auc(sorted_pairs: jnp.array) -> float:
    """
    Compute Area Under the Curve (AUC) for ROC from labels and prediction scores in sorted_pairs.

    Compared to sklearn implementation, this implementation is watered down with less options and
    eliminates most conditionals and ill-conditionals checking.

    Args:
        sorted_pairs: jnp.array
            y_true y_score pairs sorted by y_score in decreasing order,
            and it has shape n_samples * 2.
    Returns:
        roc_auc: float
    References:
        Github: scikit-learn _binary_roc_auc_score.
    """
    fpr, tpr, _ = roc_curve(sorted_pairs)
    return auc(fpr, tpr)


@jax.jit
def compute_f1_score(
    true_positive: int, false_positive: int, false_negative: int
) -> float:
    """Calculate the F1 score."""
    precision = jnp.divide(true_positive, (true_positive + false_positive))
    recall = jnp.divide(true_positive, (true_positive + false_negative))
    return jnp.divide(2 * precision * recall, (precision + recall))
