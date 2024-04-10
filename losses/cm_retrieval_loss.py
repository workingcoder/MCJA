"""MCJA/losses/cm_retrieval_loss.py
   It defines the `CMRetrievalLoss` class, a loss function specifically designed for cross-modality retrieval task.
"""

import torch
import torch.nn as nn
import torchsort

from torch.nn import functional as F


class CMRetrievalLoss(nn.Module):
    """
    A module that implements a Cross-Modal Retrieval (CMR) Loss, designed for use in training models on tasks involving
    the retrieval of relevant items across different modalities. The CMR Loss computes pairwise distances between
    embeddings from different modalities, classifying them as either matches (same identity label across modalities) or
    mismatches (different identity labels). It uses a soft ranking mechanism to assign ranks based on the pairwise
    distances, aiming to ensure that matches are ranked higher (closer) than mismatches.

    Args:
    - embeddings (Tensor): The embeddings generated by the model for a batch of items.
    - id_labels (Tensor): The identity labels for each item in the batch.
    - m_labels (Tensor): The modality labels for each item in the batch.

    Methods:
    - forward(embeddings, id_labels, m_labels): Computes the CMR Loss for a batch of embeddings,
      calculating the soft ranking loss between predicted and target ranks.
    """

    def __init__(self):
        super(CMRetrievalLoss, self).__init__()

    def forward(self, embeddings, id_labels, m_labels):
        m_labels_unique = torch.unique(m_labels)
        m_num = len(m_labels_unique)

        embeddings_list = [embeddings[m_labels == m_label] for m_label in m_labels_unique]
        id_labels_list = [id_labels[m_labels == m_label] for m_label in m_labels_unique]

        cmr_loss = 0
        valid_m_count = 0
        for i in range(m_num):
            cur_m_embeddings = embeddings_list[i]
            cur_m_id_labels = id_labels_list[i]
            other_m_embeddings = torch.cat([embeddings_list[j] for j in range(len(m_labels_unique)) if j != i], dim=0)
            other_m_id_labels = torch.cat([id_labels_list[j] for j in range(len(m_labels_unique)) if j != i], dim=0)

            match_mask = cur_m_id_labels.unsqueeze(dim=1) == other_m_id_labels.unsqueeze(dim=0)
            mismatch_mask = ~match_mask
            match_num = match_mask.sum(dim=-1)
            mismatch_num = mismatch_mask.sum(dim=-1)

            # Remove invalid queries (It has no cross-modal matching in the batch)
            remove_mask = (match_num == 0) | (match_num == len(other_m_id_labels))
            if remove_mask.all():
                continue
            cur_m_embeddings = cur_m_embeddings[~remove_mask]
            cur_m_id_labels = cur_m_id_labels[~remove_mask]
            match_mask = match_mask[~remove_mask]
            mismatch_mask = mismatch_mask[~remove_mask]
            match_num = match_num[~remove_mask]
            mismatch_num = mismatch_num[~remove_mask]

            dist_mat = F.cosine_similarity(cur_m_embeddings[:, None, :], other_m_embeddings[None, :, :], dim=-1)
            dist_mat = (1 - dist_mat) / 2

            predict_rank = torchsort.soft_rank(dist_mat, regularization="l2", regularization_strength=0.5)

            target_rank = torch.zeros_like(predict_rank)
            q_num, g_num = match_mask.shape

            target_rank[match_mask] = 1
            target_rank[mismatch_mask] = g_num

            cmr_loss += F.l1_loss(predict_rank, target_rank)

            valid_m_count += 1

        cmr_loss = cmr_loss / valid_m_count if valid_m_count > 0 else 0

        return cmr_loss
