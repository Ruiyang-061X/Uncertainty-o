import numpy as np


def compute_ece(ground_truth_list, uncertainty_list, num_bins=10):
    """
    Computes Expected Calibration Error (ECE).

    Args:
        ground_truth_list: List of 0 (correct) and 1 (incorrect) labels.
        uncertainty_list: List of uncertainty values (0.0 - 1.0).
        num_bins: Number of bins for calibration measurement.

    Returns:
        ECE score.
    """
    ground_truth_list = np.array(ground_truth_list)
    uncertainty_list = np.array(uncertainty_list)
    
    confidence_list = 1 - uncertainty_list  # Convert uncertainty to confidence

    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidence_list, bins) - 1  # Assign to bins

    ece = 0.0
    for i in range(num_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) == 0:
            continue  # Skip empty bins
        
        bin_confidence = np.mean(confidence_list[bin_mask])
        bin_accuracy = np.mean(1 - ground_truth_list[bin_mask])  # Accuracy is 1 - error rate
        bin_weight = np.sum(bin_mask) / len(confidence_list)

        ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return round(ece * 100, 1)


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
    # Compute ECE
    ece = compute_ece(ground_truth_list, uncertainty_list, num_bins=5)
    print(f"ECE Score: {ece}")