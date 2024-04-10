"""MCJA/utils/mser_rerank.py
   It introduces the Multi-Spectral Enhanced Ranking (MSER) re-ranking strategy.
"""

import numpy as np
import torch


def pairwise_distance(query_features, gallery_features):
    """
    A function that efficiently computes the pairwise Euclidean distances between two sets of features, typically used
    in the context of person re-identification tasks to measure similarities between query and gallery sets. This
    implementation leverages matrix operations for high performance, calculating the squared differences between
    each pair of features in the query and gallery feature tensors.

    Args:
    - query_features (Tensor): A tensor containing the feature vectors of the query set.
      Each row represents the feature vector of a query sample.
    - gallery_features (Tensor): A tensor containing the feature vectors of the gallery set.
      Each row represents the feature vector of a gallery sample.

    Returns:
    - Tensor: A matrix of distances where each element (i, j) represents the Euclidean distance between
      the i-th query feature and the j-th gallery feature.
    """

    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(beta=1, alpha=-2, mat1=x, mat2=y.t())
    return dist


def mser_rerank(query_feats_list, g_feats_list, k1=40, k2=20, lambda_value=0.3, eval_type=True, mode='v2i'):
    """
    A function designed to perform Multi-Spectral Enhanced Ranking (MSER) for cross-modality person re-identification.
    The MSER strategy advances the initial matching process by integrating a re-ranking mechanism that emphasizes
    reciprocal relationships and spectral characteristics. This approach recalculates the pairwise distances between
    query and gallery sets, refining the initial matches through k-reciprocal nearest neighbors and a Jaccard distance
    measure to produce a more accurate ranking.

    As mentioned in our paper, mser is based on the rerank[1] strategy in single-modality ReID and extends it to VI-ReID
    with multi-spectral information within images. Here, we utilize the rerank code from [2] in our implementation.

    Ref:
    [1] (Paper) Re-ranking Person Re-identification with k-Reciprocal Encoding, CVPR 2017.
    [2] (Code) https://github.com/DoubtedSteam/MPANet/blob/main/utils/rerank.py

    Args:
    - query_feats_list (List[Tensor]): List of tensors representing feature vectors of query images.
    - g_feats_list (List[Tensor]): List of tensors representing feature vectors of gallery images.
    - k1 (int): The primary parameter controlling the extent of k-reciprocal neighbors to consider,
      affecting the initial scope of re-ranking.
    - k2 (int): The secondary parameter influencing the expansion of reciprocal neighbors,
      further refining the selection based on mutual nearest neighbors.
    - lambda_value (float): A coefficient used to balance the original distance matrix with the Jaccard distance,
      adjusting the influence of each component in the final distance computation.
    - eval_type (bool): Indicates the type of evaluation to be performed.
    - mode (str): Specifies the modality matching direction, either 'i2v' for infrared-to-visible or 'v2i' for
      visible-to-infrared, adapting the function for different dataset characteristics.

    Returns:
    - numpy.ndarray: The re-ranked distance matrix, where each element reflects the recalculated distance between a
      query and a gallery feature vector, with lower values denoting higher similarity.
    """

    # Note: The MSER strategy requires more CPU memory.

    assert mode in ['i2v', 'v2i']

    if mode == 'i2v':
        list_num = len(g_feats_list)
        q_feat = query_feats_list[0]
        feats = torch.cat([q_feat] + g_feats_list, 0)
    else:  # mode == 'v2i'
        list_num = len(query_feats_list)
        q_feat = torch.cat(query_feats_list, 0)
        g_feat = g_feats_list[0]
        feats = torch.cat(query_feats_list + [g_feat], 0)

    dist = pairwise_distance(feats, feats)
    # dist = -torch.mm(feats, feats.permute(1, 0))
    original_dist = dist.clone().numpy()
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)

    query_num = q_feat.size(0)
    all_num = original_dist.shape[0]
    if eval_type:
        dist[:, query_num:] = dist.max()
    dist = dist.numpy()
    initial_rank = np.argsort(dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        # for j in range(len(k_reciprocal_index)):
        #     candidate = k_reciprocal_index[j]
        #     candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
        #     candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
        #                                        :int(np.around(k1 / 2)) + 1]
        #     fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
        #     candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
        #     if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
        #         candidate_k_reciprocal_index):
        #         k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []  # row index
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]

    if mode == 'i2v':
        final_dist = final_dist.reshape(query_num, list_num, -1)
        final_dist = np.mean(final_dist, axis=1)
    else:  # mode == 'v2i'
        final_dist = final_dist
        final_dist = final_dist.reshape(list_num, query_num // list_num, -1)
        final_dist = np.mean(final_dist, axis=0)
    return final_dist


if __name__ == '__main__':
    q_feat = torch.randn((8, 16))
    g_feat = torch.randn((4, 16))
    dist = mser_rerank([q_feat, q_feat, q_feat], [g_feat, g_feat, g_feat], k1=6, k2=4, mode='v2i')
    dist = mser_rerank([q_feat, q_feat, q_feat], [g_feat, g_feat, g_feat], k1=6, k2=4, mode='i2v')
