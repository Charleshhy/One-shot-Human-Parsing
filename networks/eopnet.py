import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import deeplab_xception_synBN
from sync_batchnorm import SynchronizedBatchNorm2d
from utils.util import model_mask, cross_entropy2d


class EOPNet_kway(deeplab_xception_synBN.DeepLabv3_plus_v2):
    def __init__(self, nInputChannels=3, os=16, hidden_layers=256, alpha=0.001, scaler=10., feature_lvl='high',
                 temperature=1.0, class_num=17, prototype_warmup=25):
        super(EOPNet_kway, self).__init__(
            nInputChannels=nInputChannels,
            os=os, pretrained=False)

        self.hidden_layers = hidden_layers
        self.cos_similarity_func = nn.CosineSimilarity()

        # Dynamic prototype momentum updating parameter
        self.alpha = alpha
        self.feature_lvl = feature_lvl
        self.temperature = temperature
        self.class_num = class_num
        self.prototype_warmup = prototype_warmup

        # DML layers
        self.agm_class = nn.Conv2d(hidden_layers, 2, kernel_size=1)
        self.npm_class_fg = nn.Conv2d(1, 1, kernel_size=1)
        self.npm_class_bg = nn.Conv2d(1, 1, kernel_size=1)
        self.npm_bg_fusion = nn.Conv2d(1, 1, kernel_size=1)
        self.agm_bg_fusion = nn.Conv2d(1, 1, kernel_size=1)

        # Cross entropy scaler
        self.scaler = nn.Parameter(torch.tensor(scaler), requires_grad=True)

        # Prototypes Initialization
        self.prototype = torch.nn.Parameter(torch.zeros(self.class_num, hidden_layers), requires_grad=False)
        self.s1_prototype = torch.nn.Parameter(torch.zeros(2, hidden_layers), requires_grad=False)

        # Multitask metric learning layers, s1: coarse-grained, s2: fine-grained
        self.s1_conv1 = nn.Sequential(nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1))

        self.s1_conv2 = deeplab_xception_synBN.Decoder_module(hidden_layers, hidden_layers)
        self.s1_semantic = nn.Conv2d(hidden_layers, 2, kernel_size=1)

        # Override layers
        self.decoder2 = nn.Sequential(nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1)
                                      )

        self.decoder3 = deeplab_xception_synBN.Decoder_module(hidden_layers, hidden_layers)

        # Distance layers
        self.s1_trans = nn.Conv2d(256, hidden_layers, kernel_size=1)
        self.s2_trans = nn.Conv2d(256, hidden_layers, kernel_size=1)

        self._dlab_init_weight()

    def mask2map(self, mask, class_num, ave=True):

        n, h, w = mask.shape

        maskmap_ave = torch.zeros(n, class_num, h, w).cuda()

        for i in range(class_num):
            class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
            class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
            class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)
            if ave:
                class_pix_ave = class_pix / class_sum.view(n, 1, 1)
                maskmap_ave[:, i, :, :] = class_pix_ave
            else:
                maskmap_ave[:, i, :, :] = class_pix

        return maskmap_ave

    def get_static_prototypes(self, mask, features, class_num, plobs=True):

        if plobs:
            raw_mask = torch.argmax(mask, 1)
        else:
            raw_mask = mask

        maskmap = self.mask2map(raw_mask, class_num, ave=True)
        n_batch, c_channel, h_input, w_input = features.size()

        features_ave = torch.matmul(maskmap.view(n_batch, class_num, h_input * w_input),
                                    # batch * class_num * hw
                                    features.permute(0, 2, 3, 1).view(n_batch, h_input * w_input, self.hidden_layers)
                                    # batch * hw * feature channels
                                    )  # batch * classnum * feature channels

        return features_ave

    def forward(self, input, epoch=0, cate_mapping=None):

        nclasses = self.class_num
        qry_img, s2_qry_gt_, s1_qry_gt_, sup_img, s2_sup_gt, s1_sup_gt = input

        qry_fea = self.oneshot_flex_forward(qry_img, self.feature_lvl)
        sup_fea = self.oneshot_flex_forward(sup_img, self.feature_lvl)

        qry_s1_metric, qry_s2_metric = self.s1_trans(qry_fea), self.s2_trans(qry_fea)
        sup_s1_metric, sup_s2_metric = self.s1_trans(sup_fea), self.s2_trans(sup_fea)

        # S1 cross_entropy
        s1_sup_gt = F.interpolate(s1_sup_gt, size=(sup_s1_metric.shape[2], sup_s1_metric.shape[3]), mode='nearest')
        s1_sup_sp = self.get_static_prototypes(s1_sup_gt.squeeze(1), sup_s1_metric, 2, plobs=False)
        s1_qry_out = self.s1_metric_learning(s1_sup_sp, qry_s1_metric, 2, epoch)
        s1_qry_gt = F.interpolate(s1_qry_gt_, size=(qry_s1_metric.shape[2], qry_s1_metric.shape[3]),
                                      mode='nearest')

        # S2 cross_entropy
        s2_sup_gt = F.interpolate(s2_sup_gt, size=(sup_s2_metric.shape[2], sup_s2_metric.shape[3]), mode='nearest')
        s2_sup_sp = self.get_static_prototypes(s2_sup_gt.squeeze(1), sup_s2_metric, nclasses, plobs=False)
        s2_qry_gt = F.interpolate(s2_qry_gt_, size=(qry_s2_metric.shape[2], qry_s2_metric.shape[3]), mode='nearest')
        s2_qry_sp = self.get_static_prototypes(s2_qry_gt.squeeze(1), qry_s2_metric, nclasses, plobs=False)
        
        if epoch <= self.prototype_warmup:
            pwarmup = True
        else:
            pwarmup = False

        agm_out, npm_out = self.s2_dual_metric_learning(s2_sup_sp, qry_s2_metric, pwarmup)

        distance_loss = self.prototype_wise_contrastive_learning(s1_qry_gt, s1_sup_gt, qry_s2_metric,
                                            sup_s2_metric, s2_qry_sp, s2_sup_sp)

        # Build gt by eliminating the classes that do not appear in the support set
        s2_qry_gt_ = torch.stack([model_mask(s2_qry_gt_[i], [j for j in range(nclasses) if
                                                            j not in torch.unique(s2_sup_gt[i]).tolist()])
                                   for i in range(s2_qry_gt_.shape[0])], dim=0)

        masks = {'agm_out': F.interpolate(agm_out, size=qry_img.size()[2:], mode='bilinear', align_corners=True),
                'npm_out': F.interpolate(self.scaler * npm_out, size=qry_img.size()[2:], mode='bilinear',
                                             align_corners=True),
                's2_qry_gt': s2_qry_gt_,
                 's1_qry_out': F.interpolate(s1_qry_out, size=qry_img.size()[2:], mode='bilinear', align_corners=True),
                 's1_qry_gt': s1_qry_gt_}

        losses = self.get_loss(masks, distance_loss)

        return masks, losses

    def s1_metric_learning(self, sp_ave_features, features, nclasses, pwarmup):

        # Formulate the support_features
        prototype = self.s1_prototype

        for i in range(nclasses):
            class_fea = sp_ave_features[:, i, :]
            # Generate category features and get similarity mask
            temp_list = []
            for b_ind in range(features.shape[0]):
                batch_fea = class_fea[b_ind, :]
                # When we don't want this class been parsed, proto_fea is 0
                if torch.sum(batch_fea) == 0:
                    proto_fea = torch.zeros_like(batch_fea)
                else:
                    if pwarmup or torch.sum(prototype[i]) == 0:
                        # When we want to parse this class but the prototype is 0, we use batch_fea
                        proto_fea = batch_fea.detach()
                    else:
                        # When batch_fea != 0, calculate proto_fea
                        proto_fea = (1 - self.alpha) * prototype[
                            i].clone() + self.alpha * batch_fea.detach()

                    # If training, update prototype
                    if self.training:
                        prototype[i] = proto_fea

                batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
                                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
                # print(batch_tmp_seg_map.shape)
                temp_list.append(batch_tmp_seg_map)
            # tmp_seg_map = self.cos_similarity_func(features, class_fea.unsqueeze(-1).unsqueeze(-1))
            tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
            # print(tmp_seg_map.shape)
            # bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
            # sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
            # sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

            att_fea = self.s1_conv2(self.s1_conv1(tmp_seg_map.unsqueeze(1) * features + features))
            att_mask = self.s1_semantic(att_fea)

        return att_mask

    def s2_dual_metric_learning(self, sp_ave_features, features, pwarmup):

        # This is the DML methods not using unseen class screening (ucs),
        # where the background prototype is aggregated by (sum of human cls prototypes) / (a fixed term).
        # (Variable: sup_classes_bg) is not used in this function.
        # Features initialization: AGM (denoted as att) and NCM (denoted as sim)
        npm_lis = []
        npm_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

        agm_lis = []
        agm_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

        # To reduce memory usage, we use loops to process features for each batch in each class.
        # Feel free to optimize it if your GPU has large memory
        for i in range(1, self.class_num):
            class_fea = sp_ave_features[:, i, :]

            # Generate class features and get similarity mask
            temp_list = []
            for b_ind in range(features.shape[0]):
                batch_fea = class_fea[b_ind, :]

                # When we don't want this class to be parsed, prototype for this class is set to 0
                if torch.sum(batch_fea) == 0:
                    proto_fea = torch.zeros_like(batch_fea)

                else:
                    if pwarmup or torch.sum(self.prototype[i]) == 0:
                        # When we want to parse this class but the prototype is 0, we use the batch_fea on the go
                        proto_fea = batch_fea
                    else:
                        # If batch_fea != 0, we form dymanic proto_fea
                        proto_fea = (1 - self.alpha) * self.prototype[
                            i].clone() + self.alpha * batch_fea

                    # If training, update prototype
                    if self.training:
                        self.prototype[i] = proto_fea.detach()

                batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
                                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
                temp_list.append(batch_tmp_seg_map)

            # Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
            tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
            bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
            npm_lis.append(self.npm_class_fg(tmp_seg_map.unsqueeze(1)))
            npm_bg_fea += self.npm_class_bg(bg_tmp_seg_map)

            agm_fea = self.decoder3(self.decoder2(tmp_seg_map.unsqueeze(1) * features + features))
            agm_mask = self.agm_class(agm_fea)
            agm_lis.append(agm_mask[:, 0, :, :].unsqueeze(1))
            agm_bg_fea += agm_mask[:, 1, :, :].unsqueeze(1)

        sim_bg_fea = npm_bg_fea / (self.class_num - 1)
        att_bg_fea = agm_bg_fea / (self.class_num - 1)

        npm_lis = [self.npm_bg_fusion(sim_bg_fea)] + npm_lis
        agm_lis = [self.agm_bg_fusion(att_bg_fea)] + agm_lis

        sim_mask = torch.cat(npm_lis, 1)
        att_mask = torch.cat(agm_lis, 1)

        return att_mask, sim_mask

    def contrastive_learning(self, qry_prototype, sup_prototype):

        c_qryclass, c_supclass = qry_prototype.shape[0], sup_prototype.shape[0],

        qry_nonzero = torch.nonzero(torch.sum(qry_prototype, -1)).squeeze(-1)
        sup_nonzero = torch.nonzero(torch.sum(sup_prototype, -1)).squeeze(-1)

        if qry_nonzero.shape[0] == 0:
            return torch.zeros(1).cuda()

        else:

            sim_matrix = nn.CosineSimilarity(dim=1)(qry_prototype.unsqueeze(-1),  # (qry_cls, sup_cls)
                                                    sup_prototype.permute(1, 0).unsqueeze(0)) / self.temperature
            diagonal = torch.eye(c_qryclass, c_supclass).cuda()

            sim_matrix = - diagonal[qry_nonzero, :][:, sup_nonzero].detach() * \
                         nn.LogSoftmax(dim=-1)(sim_matrix[qry_nonzero, :][:, sup_nonzero] + 10 ** -6)

            loss = torch.sum(sim_matrix, -1)

        return loss

    def get_loss(self, masks, distance_loss):

        agm_out, npm_out, s2_qry_gt, s1_qry_gt, s1_qry_pred = masks['agm_out'], masks['npm_out'], masks['s2_qry_gt'], \
                                                 masks['s1_qry_gt'], masks['s1_qry_out'],

        losses = {'agm': cross_entropy2d(agm_out, s2_qry_gt, size_average=True),
                  'npm': cross_entropy2d(npm_out, s2_qry_gt, size_average=True),
                  's1': cross_entropy2d(s1_qry_pred, s1_qry_gt, size_average=True),
                  'contrast': distance_loss}

        return losses

    def prototype_wise_contrastive_learning(self, s1_qry_gt, s1_sup_gt, qry_s2_metric,
                                            sup_s2_metric, s2_qry_sp, s2_sup_sp):

        distance_loss = 0.0

        if self.training:

            # The background area can be inferred from s1 and add to contrastive learning
            qry_prototypes_s1 = self.get_static_prototypes(s1_qry_gt.squeeze(1), qry_s2_metric, 2, plobs=False)
            sup_prototypes_s1 = self.get_static_prototypes(s1_sup_gt.squeeze(1), sup_s2_metric, 2, plobs=False)

            batch_n = s1_qry_gt.shape[0]
            qry_prototypes_lis = [torch.cat([qry_prototypes_s1[i][0].unsqueeze(0), s2_qry_sp[i, 1:, :]], 0) for i in
                                  range(batch_n)]
            sup_prototypes_lis = [torch.cat([sup_prototypes_s1[i][0].unsqueeze(0), s2_sup_sp[i, 1:, :]], 0) for i in
                                  range(batch_n)]

            distance_loss_lis = [self.contrastive_learning(qry_prototypes_lis[i], sup_prototypes_lis[i]) for i in
                                 range(batch_n)]
            distance_loss = [torch.mean(distance_loss_lis[i]) for i in range(batch_n)]
            distance_loss = torch.mean(torch.stack(distance_loss))

        return distance_loss

    def _dlab_init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EOPNet_1way(deeplab_xception_synBN.DeepLabv3_plus_v2):
    def __init__(self, nInputChannels=3, os=16, hidden_layers=256, alpha=0.001, scaler=10., feature_lvl='high',
                 temperature=1.0, class_num=17, prototype_warmup=25):
        super(EOPNet_1way, self).__init__(
            nInputChannels=nInputChannels,
            os=os, pretrained=False)

        self.hidden_layers = hidden_layers
        self.cos_similarity_func = nn.CosineSimilarity()

        # Dynamic prototype momentum updating parameter
        self.alpha = alpha
        self.feature_lvl = feature_lvl
        self.temperature = temperature
        self.class_num = class_num
        self.prototype_warmup = prototype_warmup

        # DML layers
        self.agm_class = nn.Conv2d(hidden_layers, 2, kernel_size=1)
        self.npm_class = nn.Conv2d(1, 2, kernel_size=1)

        # Cross entropy scaler
        self.scaler = nn.Parameter(torch.tensor(scaler), requires_grad=True)

        # Prototypes Initialization
        self.prototype = torch.nn.Parameter(torch.zeros(self.class_num, hidden_layers), requires_grad=False)
        self.s1_prototype = torch.nn.Parameter(torch.zeros(2, hidden_layers), requires_grad=False)

        # Multitask metric learning layers, s1: coarse-grained, s2: fine-grained
        self.s1_conv1 = nn.Sequential(nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1))

        self.s1_conv2 = deeplab_xception_synBN.Decoder_module(hidden_layers, hidden_layers)
        self.s1_semantic = nn.Conv2d(hidden_layers, 2, kernel_size=1)

        # Override layers
        self.decoder2 = nn.Sequential(nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1)
                                      )

        self.decoder3 = deeplab_xception_synBN.Decoder_module(hidden_layers, hidden_layers)

        # Distance layers
        self.s1_trans = nn.Conv2d(256, hidden_layers, kernel_size=1)
        self.s2_trans = nn.Conv2d(256, hidden_layers, kernel_size=1)

        self._dlab_init_weight()

    def mask2map(self, mask, class_num, ave=True):

        n, h, w = mask.shape

        maskmap_ave = torch.zeros(n, class_num, h, w).cuda()

        for i in range(class_num):
            class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
            class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
            class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)
            if ave:
                class_pix_ave = class_pix / class_sum.view(n, 1, 1)
                maskmap_ave[:, i, :, :] = class_pix_ave
            else:
                maskmap_ave[:, i, :, :] = class_pix

        return maskmap_ave

    def get_static_prototypes(self, mask, features, class_num, plobs=True):

        if plobs:
            raw_mask = torch.argmax(mask, 1)
        else:
            raw_mask = mask

        maskmap = self.mask2map(raw_mask, class_num, ave=True)
        n_batch, c_channel, h_input, w_input = features.size()

        features_ave = torch.matmul(maskmap.view(n_batch, class_num, h_input * w_input),
                                    # batch * class_num * hw
                                    features.permute(0, 2, 3, 1).view(n_batch, h_input * w_input, self.hidden_layers)
                                    # batch * hw * feature channels
                                    )  # batch * classnum * feature channels

        return features_ave

    def forward(self, input, epoch=0, cate_mapping=None):

        nclasses = self.class_num
        qry_img, s2_qry_gt_, s1_qry_gt_, sup_img, s2_sup_gt, s1_sup_gt = input

        qry_fea = self.oneshot_flex_forward(qry_img, self.feature_lvl)
        sup_fea = self.oneshot_flex_forward(sup_img, self.feature_lvl)

        qry_s1_metric, qry_s2_metric = self.s1_trans(qry_fea), self.s2_trans(qry_fea)
        sup_s1_metric, sup_s2_metric = self.s1_trans(sup_fea), self.s2_trans(sup_fea)

        # S1 cross_entropy
        s1_sup_gt = F.interpolate(s1_sup_gt, size=(sup_s1_metric.shape[2], sup_s1_metric.shape[3]), mode='nearest')
        s1_sup_sp = self.get_static_prototypes(s1_sup_gt.squeeze(1), sup_s1_metric, 2, plobs=False)
        s1_qry_out = self.s1_metric_learning(s1_sup_sp, qry_s1_metric, 2, epoch)

        # S2 cross_entropy
        s2_sup_gt = F.interpolate(s2_sup_gt, size=(sup_s2_metric.shape[2], sup_s2_metric.shape[3]), mode='nearest')
        s2_sup_sp = self.single_cls_feature(sup_s2_metric, s2_sup_gt)

        if epoch <= self.prototype_warmup:
            pwarmup = True
        else:
            pwarmup = False

        agm_out, npm_out = self.s2_dual_metric_1way(s2_sup_sp, qry_s2_metric, pwarmup, cate_mapping=cate_mapping)

        # Build gt by eliminating the classes that do not appear in the support set
        s2_qry_gt_ = torch.stack([model_mask(s2_qry_gt_[i], [j for j in range(nclasses) if
                                                             j not in torch.unique(s2_sup_gt[i]).tolist()])
                                  for i in range(s2_qry_gt_.shape[0])], dim=0)

        masks = {'agm_out': F.interpolate(agm_out, size=qry_img.size()[2:], mode='bilinear', align_corners=True),
                 'npm_out': F.interpolate(self.scaler * npm_out, size=qry_img.size()[2:], mode='bilinear',
                                          align_corners=True),
                 's2_qry_gt': s2_qry_gt_,
                 's1_qry_out': F.interpolate(s1_qry_out, size=qry_img.size()[2:], mode='bilinear', align_corners=True),
                 's1_qry_gt': s1_qry_gt_}

        losses = self.get_loss(masks, torch.zeros(1).to(s2_qry_gt_.device))

        return masks, losses

    def s1_metric_learning(self, sp_ave_features, features, nclasses, pwarmup):

        # Formulate the support_features
        prototype = self.s1_prototype

        for i in range(nclasses):
            class_fea = sp_ave_features[:, i, :]
            # Generate category features and get similarity mask
            temp_list = []
            for b_ind in range(features.shape[0]):
                batch_fea = class_fea[b_ind, :]
                # When we don't want this class been parsed, proto_fea is 0
                if torch.sum(batch_fea) == 0:
                    proto_fea = torch.zeros_like(batch_fea)
                else:
                    if pwarmup or torch.sum(prototype[i]) == 0:
                        # When we want to parse this class but the prototype is 0, we use batch_fea
                        proto_fea = batch_fea.detach()
                    else:
                        # When batch_fea != 0, calculate proto_fea
                        proto_fea = (1 - self.alpha) * prototype[
                            i].clone() + self.alpha * batch_fea.detach()

                    # If training, update prototype
                    if self.training:
                        prototype[i] = proto_fea

                batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
                                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
                # print(batch_tmp_seg_map.shape)
                temp_list.append(batch_tmp_seg_map)
            # tmp_seg_map = self.cos_similarity_func(features, class_fea.unsqueeze(-1).unsqueeze(-1))
            tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
            # print(tmp_seg_map.shape)
            # bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
            # sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
            # sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

            att_fea = self.s1_conv2(self.s1_conv1(tmp_seg_map.unsqueeze(1) * features + features))
            att_mask = self.s1_semantic(att_fea)

        return att_mask

    def single_cls_feature(self, features, mask):
        batch_n, _, mask_w, mask_h = features.size()
        pos_sum = torch.sum(
            mask.view(batch_n, mask_h * mask_w), dim=1).unsqueeze(1)
        pos_sum = torch.where(pos_sum == 0, torch.ones(1).cuda(), pos_sum)
        vec_pos = torch.sum(torch.sum(features * mask, dim=3), dim=2) / pos_sum

        return vec_pos

    def s2_dual_metric_1way(self, sp_ave_features, features, pwarmup, cate_mapping):
        # Return att_mask and sim_mask
        # Generate category features and get similarity mask
        temp_list = []
        for b_ind in range(features.shape[0]):
            batch_fea = sp_ave_features[b_ind, :]
            i = cate_mapping[b_ind]

            # When we don't want this class to be parsed, prototype for this class is set to 0
            if torch.sum(batch_fea) == 0:
                proto_fea = torch.zeros_like(batch_fea)

            else:
                if pwarmup or torch.sum(self.prototype[i]) == 0:
                    # When we want to parse this class but the prototype is 0, we use the batch_fea on the go
                    proto_fea = batch_fea
                else:
                    # If batch_fea != 0, we form dymanic proto_fea
                    proto_fea = (1 - self.alpha) * self.prototype[
                        i].clone() + self.alpha * batch_fea

                # If training, update prototype
                if self.training:
                    self.prototype[i] = proto_fea.detach()

            batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
                                                         proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
            temp_list.append(batch_tmp_seg_map)

        # Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
        tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
        tmp_seg = tmp_seg_map.unsqueeze(dim=1)
        res_features = features * tmp_seg
        res_features = self.decoder3(self.decoder2(res_features))
        npm_mask = self.npm_class(tmp_seg)
        agm_mask = self.agm_class(res_features)

        return agm_mask, npm_mask

    def contrastive_learning(self, qry_prototype, sup_prototype):

        c_qryclass, c_supclass = qry_prototype.shape[0], sup_prototype.shape[0],

        qry_nonzero = torch.nonzero(torch.sum(qry_prototype, -1)).squeeze(-1)
        sup_nonzero = torch.nonzero(torch.sum(sup_prototype, -1)).squeeze(-1)

        if qry_nonzero.shape[0] == 0:
            return torch.zeros(1).cuda()

        else:

            sim_matrix = nn.CosineSimilarity(dim=1)(qry_prototype.unsqueeze(-1),  # (qry_cls, sup_cls)
                                                    sup_prototype.permute(1, 0).unsqueeze(0)) / self.temperature
            diagonal = torch.eye(c_qryclass, c_supclass).cuda()

            sim_matrix = - diagonal[qry_nonzero, :][:, sup_nonzero].detach() * \
                         nn.LogSoftmax(dim=-1)(sim_matrix[qry_nonzero, :][:, sup_nonzero] + 10 ** -6)

            loss = torch.sum(sim_matrix, -1)

        return loss

    def get_loss(self, masks, distance_loss):

        agm_out, npm_out, s2_qry_gt, s1_qry_gt, s1_qry_pred = masks['agm_out'], masks['npm_out'], masks['s2_qry_gt'], \
                                                              masks['s1_qry_gt'], masks['s1_qry_out'],

        losses = {'agm': cross_entropy2d(agm_out, s2_qry_gt, size_average=True),
                  'npm': cross_entropy2d(npm_out, s2_qry_gt, size_average=True),
                  's1': cross_entropy2d(s1_qry_pred, s1_qry_gt, size_average=True),
                  'contrast': distance_loss}

        return losses

    def prototype_wise_contrastive_learning(self, s1_qry_gt, s1_sup_gt, qry_s2_metric,
                                            sup_s2_metric, s2_qry_sp, s2_sup_sp):

        distance_loss = 0.0

        if self.training:
            # The background area can be inferred from s1 and add to contrastive learning
            qry_prototypes_s1 = self.get_static_prototypes(s1_qry_gt.squeeze(1), qry_s2_metric, 2, plobs=False)
            sup_prototypes_s1 = self.get_static_prototypes(s1_sup_gt.squeeze(1), sup_s2_metric, 2, plobs=False)

            batch_n = s1_qry_gt.shape[0]
            qry_prototypes_lis = [torch.cat([qry_prototypes_s1[i][0].unsqueeze(0), s2_qry_sp[i, 1:, :]], 0) for i in
                                  range(batch_n)]
            sup_prototypes_lis = [torch.cat([sup_prototypes_s1[i][0].unsqueeze(0), s2_sup_sp[i, 1:, :]], 0) for i in
                                  range(batch_n)]

            distance_loss_lis = [self.contrastive_learning(qry_prototypes_lis[i], sup_prototypes_lis[i]) for i in
                                 range(batch_n)]
            distance_loss = [torch.mean(distance_loss_lis[i]) for i in range(batch_n)]
            distance_loss = torch.mean(torch.stack(distance_loss))

        return distance_loss

    def _dlab_init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()