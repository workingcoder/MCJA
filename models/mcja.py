"""MCJA/models/mcja.py
   It defines the Multi-level Cross-modality Joint Alignment (MCJA) model, a framework for cross-modality VI-ReID task.
"""

import torch
import torch.nn as nn

from models.backbones.resnet import resnet50
from models.modules.mda import MDA
from losses.cm_retrieval_loss import CMRetrievalLoss
from utils.calc_acc import calc_acc


class MCJA(nn.Module):
    """
    The Class of Multi-Channel Joint Analysis (MCJA) model, designed for cross-modality person re-identification tasks.
    This model integrates various components, including a backbone for feature extraction, the Modality Distribution
    Adapter (MDA) for better cross-modality feature alignment & distribution adaptation, a neck for feature embedding,
    a head for classification, and specialized loss functions (identity and cross-modality retrieval (CMR) losses).

    Args:
    - num_classes (int): The number of identity classes in the dataset.
    - drop_last_stride (bool): A flag to determine whether the last stride in the backbone should be dropped.
    - mda_ratio (int): The ratio for reducing the channel dimensions in MDA layers.
    - mda_m (int): The number of modalities considered by the MDA layers.
    - loss_id (bool): Whether to use the identity loss during training.
    - loss_cmr (bool): Whether to use the cross-modality retrieval loss during training.

    Methods:
    - forward(inputs, labels=None, **kwargs): Processes the input through the MCJA model.
      In training mode, it computes the loss and metrics based on the provided labels and additional information (e.g.,
      camera IDs for modality labels). In evaluation mode, it returns the feature embeddings after BN neck processing.
    - train_forward(embeddings, labels, **kwargs): A helper function called during training to compute losses.
      It calculates the identity and CMR losses based on embeddings, identity labels, and modality labels.
    """

    def __init__(self, num_classes, drop_last_stride=False, mda_ratio=2, mda_m=2, loss_id=True, loss_cmr=True):
        super(MCJA, self).__init__()

        # Backbone -----------------------------------------------------------------------------------------------------
        self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
        self.base_dim = 2048

        # Neck ---------------------------------------------------------------------------------------------------------
        self.bn_neck = nn.BatchNorm1d(self.base_dim)
        nn.init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)

        # Head ---------------------------------------------------------------------------------------------------------
        self.classifier = nn.Linear(self.base_dim, num_classes, bias=False)

        # Loss ---------------------------------------------------------------------------------------------------------
        self.id_loss = nn.CrossEntropyLoss() if loss_id else None
        ###### Cross-Modality Retrieval Loss (CMR) ######
        self.cmr_loss = CMRetrievalLoss() if loss_cmr else None

        # Module -------------------------------------------------------------------------------------------------------
        layers = [3, 4, 6, 3]  # Just for ResNet50
        ###### Modality Distribution Adapter (MDA) ######
        mda_layers = [0, 2, 3, 0]
        self.MDA_1 = nn.ModuleList(
            [MDA(in_channels=256, inter_ratio=mda_ratio, m_num=mda_m) for _ in range(mda_layers[0])])
        self.MDA_1_idx = sorted([layers[0] - (i + 1) for i in range(mda_layers[0])])
        self.MDA_2 = nn.ModuleList(
            [MDA(in_channels=512, inter_ratio=mda_ratio, m_num=mda_m) for _ in range(mda_layers[1])])
        self.MDA_2_idx = sorted([layers[1] - (i + 1) for i in range(mda_layers[1])])
        self.MDA_3 = nn.ModuleList(
            [MDA(in_channels=1024, inter_ratio=mda_ratio, m_num=mda_m) for _ in range(mda_layers[2])])
        self.MDA_3_idx = sorted([layers[2] - (i + 1) for i in range(mda_layers[2])])
        self.MDA_4 = nn.ModuleList(
            [MDA(in_channels=2048, inter_ratio=mda_ratio, m_num=mda_m) for _ in range(mda_layers[3])])
        self.MDA_4_idx = sorted([layers[3] - (i + 1) for i in range(mda_layers[3])])

    def forward(self, inputs, labels=None, **kwargs):

        # Feature Extraction -------------------------------------------------------------------------------------------
        feats = self.backbone.conv1(inputs)
        feats = self.backbone.bn1(feats)
        feats = self.backbone.relu(feats)
        feats = self.backbone.maxpool(feats)

        MDA_1_counter = 0
        if len(self.MDA_1_idx) == 0: self.MDA_1_idx = [-1]
        for i in range(len(self.backbone.layer1)):
            feats = self.backbone.layer1[i](feats)
            if i == self.MDA_1_idx[MDA_1_counter]:
                _, C, H, W = feats.shape
                feats = self.MDA_1[MDA_1_counter](feats)
                MDA_1_counter += 1
        MDA_2_counter = 0
        if len(self.MDA_2_idx) == 0: self.MDA_2_idx = [-1]
        for i in range(len(self.backbone.layer2)):
            feats = self.backbone.layer2[i](feats)
            if i == self.MDA_2_idx[MDA_2_counter]:
                _, C, H, W = feats.shape
                feats = self.MDA_2[MDA_2_counter](feats)
                MDA_2_counter += 1
        MDA_3_counter = 0
        if len(self.MDA_3_idx) == 0: self.MDA_3_idx = [-1]
        for i in range(len(self.backbone.layer3)):
            feats = self.backbone.layer3[i](feats)
            if i == self.MDA_3_idx[MDA_3_counter]:
                _, C, H, W = feats.shape
                feats = self.MDA_3[MDA_3_counter](feats)
                MDA_3_counter += 1
        MDA_4_counter = 0
        if len(self.MDA_4_idx) == 0: self.MDA_4_idx = [-1]
        for i in range(len(self.backbone.layer4)):
            feats = self.backbone.layer4[i](feats)
            if i == self.MDA_4_idx[MDA_4_counter]:
                _, C, H, W = feats.shape
                feats = self.MDA_4[MDA_4_counter](feats)
                MDA_4_counter += 1
        global_feats = feats

        # Feature Embedding --------------------------------------------------------------------------------------------
        b, c, h, w = global_feats.shape
        global_feats = global_feats.view(b, c, -1)
        p = 3.0
        embeddings = (torch.mean(global_feats ** p, dim=-1) + 1e-12) ** (1 / p)  # GeMPooling

        # Train & Test Return ------------------------------------------------------------------------------------------
        if self.training:
            return self.train_forward(embeddings, labels, **kwargs)
        else:
            return self.bn_neck(embeddings)

    def train_forward(self, embeddings, labels, **kwargs):
        loss = 0
        metric = {}

        embeddings = self.bn_neck(embeddings)

        # modality labels
        cam_ids = kwargs.get('cam_ids')
        rgb_idx_mask = (cam_ids == 1) + (cam_ids == 2) + (cam_ids == 4) + (cam_ids == 5)
        ir_idx_mask = (cam_ids == 3) + (cam_ids == 6)
        m_labels = torch.ones((len(labels)))
        m_labels[rgb_idx_mask] = 0
        m_labels[ir_idx_mask] = 1

        if self.cmr_loss is not None:
            ###### Cross-Modality Retrieval Loss (CMR) ######
            cmr_loss = self.cmr_loss(embeddings.float(), id_labels=labels, m_labels=m_labels)
            loss += cmr_loss
            metric.update({'loss_cmr': cmr_loss.data})

        logits = self.classifier(embeddings)

        if self.id_loss is not None:
            # Identity Loss (ID Loss)
            id_loss = self.id_loss(logits.float(), labels)
            loss += id_loss
            metric.update({'cls_acc': calc_acc(logits.data, labels), 'loss_id': id_loss.data})

        return loss, metric
