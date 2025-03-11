import numpy as np


def compute_aurac(ground_truth_list, uncertainty_list):
    """
    Computes AURAC: Area Under the Risk-Accuracy Curve.

    Args:
        ground_truth_list: List of 0 (correct) and 1 (incorrect) labels.
        uncertainty_list: List of uncertainty values (0.0 - 1.0).

    Returns:
        AURAC score.
    """
    sorted_indices = np.argsort(uncertainty_list)  # Sort by uncertainty (low to high)
    sorted_truths = np.array(ground_truth_list)[sorted_indices]

    risk = np.cumsum(sorted_truths) / np.arange(1, len(sorted_truths) + 1)  # Cumulative error rate
    accuracy = 1 - risk  # Accuracy = 1 - risk

    # Compute AURAC using trapezoidal rule
    aurac = np.trapz(accuracy, dx=1.0 / len(accuracy))
    return round(aurac * 100, 1)


if __name__ == "__main__":
    # Sample ground truth (0: correct, 1: incorrect) and uncertainty values (0.0 - 1.0)
    ground_truth_list = [
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
    uncertainty_list = [
        0.6554587535412855,
        0.0,
        0.31091750708257115,
        0.31091750708257115,
        0.4181656600790516,
        0.0,
        0.0,
        0.4181656600790516,
        0.0,
        0.0,
        0.4181656600790516,
        0.31091750708257115,
        0.5904362833084089,
        0.31091750708257115,
        0.0,
        0.31091750708257115,
        0.31091750708257115,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.31091750708257115,
        0.0,
        0.0,
        0.5904362833084089,
        0.31091750708257115,
        0.6554587535412855,
        0.4181656600790516,
        0.31091750708257115,
        0.0,
        0.0,
        0.4181656600790516,
        0.0,
        0.31091750708257115,
        0.31091750708257115,
        0.0,
        0.0,
        0.4181656600790516,
        0.0,
        0.31091750708257115,
        0.31091750708257115,
        0.31091750708257115,
        0.0,
        0.31091750708257115,
        0.31091750708257115,
        0.31091750708257115,
        0.0,
        0.4181656600790516,
        0.31091750708257115,
        0.0,
        0.31091750708257115,
        0.31091750708257115,
        0.0,
        0.5904362833084089,
        0.5904362833084089,
        0.31091750708257115,
        0.0,
        0.0,
        0.0
    ]
    # Compute AURAC
    aurac = compute_aurac(ground_truth_list, uncertainty_list)
    print(f"AURAC Score: {aurac}")