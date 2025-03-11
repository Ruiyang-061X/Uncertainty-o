from sklearn.metrics import roc_auc_score


def compute_auroc(ground_truth_list, uncertainty_list):
    """
    Computes AUROC: Measures how well uncertainty values distinguish between correct and incorrect predictions.

    Args:
        ground_truth_list: List of 0 (correct) and 1 (incorrect) labels.
        uncertainty_list: List of uncertainty values (0.0 - 1.0).

    Returns:
        AUROC score.
    """
    auroc = roc_auc_score(ground_truth_list, uncertainty_list)
    return round(auroc * 100.0, 1)


if __name__ == "__main__":
    # Sample ground truth (0: correct, 1: incorrect) and uncertainty values (0.0 - 1.0)
    # ground_truth_list = [0,      1,    0,    1,    1,    0,    0,     1,     0,      1]
    # uncertainty_list =  [0.1,  0.8,  0.2,  0.9,  0.7,  0.3,  0.2,  0.85,  0.15,  0.95]

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

    # Compute AUROC
    auroc = compute_auroc(ground_truth_list, uncertainty_list)
    print(f"AUROC Score: {auroc}")