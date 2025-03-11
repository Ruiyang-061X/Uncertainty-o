import collections
import math


def uncertainty_caculation(cluster_idx_list):
    N = len(cluster_idx_list)
    cluster_dis = collections.Counter(cluster_idx_list)
    uncertainty = -sum((cnt / N) * math.log2(cnt / N) for cnt in cluster_dis.values())
    idx_list_max_uncertainty = list(range(N))
    cluster_dis_max_uncertainty = collections.Counter(idx_list_max_uncertainty)
    max_uncertainty = -sum((cnt / N) * math.log2(cnt / N) for cnt in cluster_dis_max_uncertainty.values())
    uncertainty = uncertainty / max_uncertainty
    return uncertainty


if __name__ == "__main__":
    cluster_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    uncertainty = uncertainty_caculation(cluster_idx_list)
    print(cluster_idx_list)
    print(uncertainty)
    print('-' * 100)
    cluster_idx_list = [0, 0, 0]
    uncertainty = uncertainty_caculation(cluster_idx_list)
    print(cluster_idx_list)
    print(uncertainty)