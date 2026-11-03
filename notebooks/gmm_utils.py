#!/usr/bin/env python3
from sklearn import metrics
import numpy as np


def compute_metrics(labels_true, labels_pred):
    metrics_res = {}
    metrics_res["balanced_accuracy_score"] = metrics.balanced_accuracy_score(
        y_true=labels_true, y_pred=labels_pred
    )
    metrics_res["cohen_kappa_score"] = metrics.cohen_kappa_score(
        y1=labels_true, y2=labels_pred
    )
    # metrics_res["hinge_loss"] = metrics.hinge_loss(y_true=labels_true, pred_decision=labels_pred)
    metrics_res["matthews_corrcoef"] = metrics.matthews_corrcoef(
        y_true=labels_true, y_pred=labels_pred
    )
    # metrics_res["roc_auc_score"] = metrics.roc_auc_score(y_true=labels_true, y_score=labels_pred, multi_class="ovo", average="macro")

    for m_key, m_val in metrics_res.items():
        print(f"{m_key}: {m_val}")
    return metrics_res


def features_augment(df, window_size, min_periods):
    skip_col = ["label_true"]
    freeze_cols = df.columns
    return df, freeze_cols
    for col in freeze_cols:
        if col in skip_col:
            continue
        df[f"{col}_roll_mean"] = df[col].rolling(window=window_size).mean()
        df[f"{col}_roll_med"] = df[col].rolling(window=window_size).median()
        df[f"{col}_roll_max"] = df[col].rolling(window=window_size).max()
        # df[f"{col}_roll_min"] = df[col].rolling(window=window_size).min()
        df[f"{col}_roll_std"] = df[col].rolling(window=window_size).std()
        df[f"{col}_roll_25p"] = df[col].rolling(window=window_size).quantile(0.25)
        df[f"{col}_roll_75p"] = df[col].rolling(window=window_size).quantile(0.75)
        df[f"{col}_exp_mean"] = df[col].expanding(min_periods=min_periods).mean()
        df[f"{col}_exp_med"] = df[col].expanding(min_periods=min_periods).median()
        # df[f"{col}_exp_max"] = df[col].expanding(min_periods=min_periods).max()
        # df[f"{col}_exp_min"] = df[col].expanding(min_periods=min_periods).min()
        df[f"{col}_exp_std"] = df[col].expanding(min_periods=min_periods).std()
        df[f"{col}_exp_25p"] = df[col].expanding(min_periods=min_periods).quantile(0.25)
        df[f"{col}_exp_75p"] = df[col].expanding(min_periods=min_periods).quantile(0.75)
    df = df.fillna(0)
    return df, freeze_cols


def select_supervised_data(total_sample_size, labels_gt, K_components, U):
    unique_labels = np.unique(labels_gt)
    label_sample_size = int(np.round(total_sample_size / len(unique_labels)))

    # Select balanced data for each label
    all_idx = np.arange(labels_gt.shape[0])
    idx_superv = []
    for label in unique_labels:
        if np.isnan(label):
            continue
        current_label_mask = labels_gt == label
        current_label_idx = np.array(
            [idx for idx in all_idx if current_label_mask[idx]]
        )
        assert len(current_label_idx) >= label_sample_size, (
            f"Not enough data points to select {label_sample_size} samples for class {label}, only {len(current_label_idx)} found"
        )
        idx_superv.extend(
            np.random.choice(current_label_idx, size=label_sample_size, replace=False)
        )
    idx_superv = np.array(idx_superv)

    # Build the Y matrix
    Y = np.zeros((len(idx_superv), K_components)).astype(int)
    for idxi, idx_val in enumerate(idx_superv):
        Y[idxi, int(labels_gt[int(idx_val)])] = 1

    X = U[idx_superv, :]
    return X, Y, idx_superv


def flat_labels2binary_mat(flat_labels):
    dim_cols = len(np.unique(flat_labels))
    dim_rows = len(flat_labels)
    labels_mat = np.zeros(shape=(dim_rows, dim_cols)).astype(int)
    for lab_i, lab_val in enumerate(flat_labels):
        labels_mat[lab_i, int(lab_val)] = 1
    return labels_mat
