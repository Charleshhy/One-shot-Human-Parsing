import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import deeplab_xception_synBN


# For code release
def model_mask(mask, delete_class_set):
	new_mask = mask.clone()
	for i in delete_class_set:
		new_mask = torch.where(mask == i, torch.zeros(1).cuda(), new_mask)

	return new_mask


class popnet_kway_dp(
	deeplab_xception_synBN.DeepLabv3_plus_v2):
	def __init__(self, nInputChannels=3, n_classes=7, os=16, hidden_layers=256, beta=0.001, scaler=10.,
	             feature_lvl='high', DML_mode='fixed'):
		super(popnet_kway_dp, self).__init__(nInputChannels=nInputChannels,
		                                     n_classes=n_classes,
		                                     os=os, pretrained=True)

		# Settings
		self.hidden_layers = hidden_layers
		self.DML_mode = DML_mode
		self.cos_similarity_func = nn.CosineSimilarity()
		self.feature_lvl = feature_lvl
		self.beta = beta

		# AGM layers
		self.classifier_6 = nn.Conv2d(256, 2, kernel_size=1)
		self.bg_att_fusion = nn.Conv2d(1, 1, kernel_size=1)

		# NCM layers
		self.after_sim_fg = nn.Conv2d(1, 1, kernel_size=1)
		self.after_sim_bg = nn.Conv2d(1, 1, kernel_size=1)
		self.bg_sim_fusion = nn.Conv2d(1, 1, kernel_size=1)
		self.scaler = nn.Parameter(torch.tensor(scaler), requires_grad=True)  # Learnable scaler

		# KIM layers
		self.feature_fusion = nn.Sequential(
			deeplab_xception_synBN.Decoder_module(512, 256),
			deeplab_xception_synBN.Decoder_module(256, 256),
		)

		self.prototype = torch.nn.Parameter(torch.zeros(n_classes, 256), requires_grad=False)

	def mask2map(self, mask, class_num):
		# Helper function for getting feature indexes for each class (gpu)

		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w).cuda()

		for i in range(class_num):
			class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			maskmap_ave[:, i, :, :] = class_pix_ave

		return maskmap_ave

	def mask2map_cpu(self, mask, class_num):
		# Helper function for getting feature indexes for each class
		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w)

		for i in range(class_num):
			class_pix = torch.where(mask == i, torch.ones(1), torch.zeros(1))
			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			class_sum = torch.where(class_sum == 0, torch.ones(1), class_sum)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			maskmap_ave[:, i, :, :] = class_pix_ave

		return maskmap_ave

	def forward(self, input, cate_num=17, proto_prev_stage=True, prev_qry_fea=None, prev_sup_fea=None):

		nclasses = cate_num
		img, support, support_mask = input

		# Encoder
		img_features = self.oneshot_flex_forward(img, feature_lvl=self.feature_lvl)
		sup_features = self.oneshot_flex_forward(support, feature_lvl=self.feature_lvl)

		# Knowledge infusion module
		if prev_qry_fea is not None and prev_sup_fea is not None:
			if prev_qry_fea.shape != img_features.shape:
				prev_qry_fea = F.upsample(prev_qry_fea, size=img_features.size()[2:], mode='bilinear',
				                          align_corners=True)
				prev_sup_fea = F.upsample(prev_sup_fea, size=sup_features.size()[2:], mode='bilinear',
				                          align_corners=True)

			img_features = self.feature_fusion(torch.cat([prev_qry_fea, img_features], dim=1))
			sup_features = self.feature_fusion(torch.cat([prev_sup_fea, sup_features], dim=1))

		batch_n, _, mask_h, mask_w = sup_features.size()
		support_mask = F.upsample(support_mask, size=(mask_h, mask_w), mode='nearest')

		# Get indexes for each class
		maskmap = self.mask2map(support_mask.squeeze(1), nclasses)

		# Compute average features using the indexes
		sp_ave_features = torch.matmul(maskmap.view(batch_n, nclasses, mask_h * mask_w),
		                               # batch * class_num * hw
		                               sup_features.permute(0, 2, 3, 1).view(batch_n, mask_h * mask_w,
		                                                                     self.hidden_layers)
		                               # batch * hw * feature channels
		                               )  # batch * classnum * feature channels

		sup_classes = torch.unique(support_mask).long()

		if self.DML_mode == 'fixed':
			dml = self.dual_metric_fixed
		else:
			dml = self.dual_metric_ucs

		att_mask, sim_mask = dml(sp_ave_features, img_features, nclasses, proto_prev_stage, sup_classes)

		return F.upsample(att_mask, size=img.size()[2:], mode='bilinear', align_corners=True), F.upsample(
			self.scaler * sim_mask, size=img.size()[2:], mode='bilinear', align_corners=True)

	def dual_metric_ucs(self, sp_ave_features, features, nclasses, proto_prev_stage, sup_classes_bg):
		# This is the DML methods using unseen class screening (ucs),
		# where the background prototype is aggregated by
		# (sum human cls prototypes) / (# of human cls prototypes annotated in support image).

		# Get rid of the background index and remain the existing indexes
		# We only calculate
		max_classes = nclasses
		sup_classes = sup_classes_bg[sup_classes_bg != 0]
		screened_sp = sp_ave_features[:, sup_classes, :]

		# Features initialization: AGM (denoted as att) and NCM (denoted as sim)
		sem_lis = []
		sim_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		att_lis = []
		att_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		# To reduce memory usage, we use loops to process features for each batch in each class
		for i in range(len(sup_classes)):

			class_fea = screened_sp[:, i, :]

			# Generate class features and get similarity mask
			temp_list = []
			for b_ind in range(features.shape[0]):
				batch_fea = class_fea[b_ind, :]

				# When we don't want this class to be parsed, prototype for this class is set to 0
				if torch.sum(batch_fea) == 0:
					proto_fea = torch.zeros_like(batch_fea)

				else:
					if proto_prev_stage or torch.sum(self.prototype[i]) == 0:
						# When we want to parse this class but the prototype is 0, we use the batch_fea on the go
						proto_fea = batch_fea
					else:
						proto_num = sup_classes[i]
						if proto_prev_stage or torch.sum(self.prototype[proto_num]) == 0:
							# When we want to parse this class but the prototype is 0, we use batch_fea
							proto_fea = batch_fea
						else:
							# When batch_fea != 0, calculate proto_fea
							proto_fea = (1 - self.beta) * self.prototype[
								proto_num].clone() + self.beta * batch_fea

					# If training, update prototype
					if self.training:
						self.prototype[i] = proto_fea.detach()

				batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
				                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
				temp_list.append(batch_tmp_seg_map)

			# Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
			tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
			bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
			sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
			sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

			att_fea = self.decoder3(self.decoder2(tmp_seg_map.unsqueeze(1) * features + features))
			att_mask = self.classifier_6(att_fea)
			att_lis.append(att_mask[:, 0, :, :].unsqueeze(1))
			att_bg_fea += att_mask[:, 1, :, :].unsqueeze(1)

		if len(sup_classes) != 0:
			sim_bg_fea = sim_bg_fea / len(sup_classes)
			att_bg_fea = att_bg_fea / len(sup_classes)
		else:
			sim_bg_fea = torch.ones_like(sim_bg_fea).cuda()
			att_bg_fea = torch.ones_like(att_bg_fea).cuda()

		sem_lis = [self.bg_sim_fusion(sim_bg_fea)] + sem_lis
		att_lis = [self.bg_att_fusion(att_bg_fea)] + att_lis

		# bg has to be added
		sim_mask_screened = torch.cat(sem_lis, 1)
		att_mask_screened = torch.cat(att_lis, 1)

		sim_mask, att_mask = torch.zeros(features.shape[0], max_classes, features.shape[2], features.shape[3]).cuda(), \
		                     torch.zeros(features.shape[0], max_classes, features.shape[2], features.shape[3]).cuda()

		# Follow the previous indexes
		sim_mask[:, sup_classes_bg], att_mask[:, sup_classes_bg] = sim_mask_screened, att_mask_screened

		return att_mask, sim_mask

	def dual_metric_fixed(self, sp_ave_features, features, nclasses, proto_prev_stage, sup_classes_bg):

		# This is the DML methods not using unseen class screening (ucs),
		# where the background prototype is aggregated by (sum of human cls prototypes) / (a fixed term).
		# (Variable: sup_classes_bg) is not used in this function.

		# Features initialization: AGM (denoted as att) and NCM (denoted as sim)
		sem_lis = []
		sim_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		att_lis = []
		att_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		# To reduce memory usage, we use loops to process features for each batch in each class
		for i in range(1, nclasses):

			class_fea = sp_ave_features[:, i, :]

			# Generate class features and get similarity mask
			temp_list = []
			for b_ind in range(features.shape[0]):
				batch_fea = class_fea[b_ind, :]

				# When we don't want this class to be parsed, prototype for this class is set to 0
				if torch.sum(batch_fea) == 0:
					proto_fea = torch.zeros_like(batch_fea)

				else:
					if proto_prev_stage or torch.sum(self.prototype[i]) == 0:
						# When we want to parse this class but the prototype is 0, we use the batch_fea on the go
						proto_fea = batch_fea
					else:
						# If batch_fea != 0, we form dymanic proto_fea
						proto_fea = (1 - self.beta) * self.prototype[
							i].clone() + self.beta * batch_fea

					# If training, update prototype
					if self.training:
						self.prototype[i] = proto_fea.detach()

				batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
				                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
				temp_list.append(batch_tmp_seg_map)

			# Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
			tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
			bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
			sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
			sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

			att_fea = self.decoder3(self.decoder2(tmp_seg_map.unsqueeze(1) * features + features))
			att_mask = self.classifier_6(att_fea)
			att_lis.append(att_mask[:, 0, :, :].unsqueeze(1))
			att_bg_fea += att_mask[:, 1, :, :].unsqueeze(1)

		sim_bg_fea = sim_bg_fea / (nclasses - 1)
		att_bg_fea = att_bg_fea / (nclasses - 1)

		sem_lis = [self.bg_sim_fusion(sim_bg_fea)] + sem_lis
		att_lis = [self.bg_att_fusion(att_bg_fea)] + att_lis

		sim_mask = torch.cat(sem_lis, 1)
		att_mask = torch.cat(att_lis, 1)

		return att_mask, sim_mask


class popnet_kway_dp_more_dataset(
	deeplab_xception_synBN.DeepLabv3_plus_v2):
	def __init__(self, nInputChannels=3, os=16, hidden_layers=256, beta=0.001, scaler=10.,
	             feature_lvl='high', DML_mode='fixed', cate_num=17):
		super(popnet_kway_dp_more_dataset, self).__init__(nInputChannels=nInputChannels,
		                                                  n_classes=cate_num,
		                                                  os=os, pretrained=True)

		# Settings
		self.hidden_layers = hidden_layers
		self.DML_mode = DML_mode
		self.cos_similarity_func = nn.CosineSimilarity()
		self.feature_lvl = feature_lvl
		self.beta = beta
		self.cate_num = cate_num

		# AGM layers
		self.classifier_6 = nn.Conv2d(256, 2, kernel_size=1)
		self.bg_att_fusion = nn.Conv2d(1, 1, kernel_size=1)

		# NCM layers
		self.after_sim_fg = nn.Conv2d(1, 1, kernel_size=1)
		self.after_sim_bg = nn.Conv2d(1, 1, kernel_size=1)
		self.bg_sim_fusion = nn.Conv2d(1, 1, kernel_size=1)
		self.scaler = nn.Parameter(torch.tensor(scaler), requires_grad=True)  # Learnable scaler

		# KIM layers
		self.feature_fusion = nn.Sequential(
			deeplab_xception_synBN.Decoder_module(512, 256),
			deeplab_xception_synBN.Decoder_module(256, 256),
		)

		self.prototype = torch.nn.Parameter(torch.zeros(cate_num, 256), requires_grad=False)

	def mask2map(self, mask, class_num):
		# Helper function for getting feature indexes for each class (gpu)

		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w).cuda()

		for i in range(class_num):
			class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			maskmap_ave[:, i, :, :] = class_pix_ave

		return maskmap_ave

	def mask2map_cpu(self, mask, class_num):
		# Helper function for getting feature indexes for each class
		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w)

		for i in range(class_num):
			class_pix = torch.where(mask == i, torch.ones(1), torch.zeros(1))
			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			class_sum = torch.where(class_sum == 0, torch.ones(1), class_sum)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			maskmap_ave[:, i, :, :] = class_pix_ave

		return maskmap_ave

	def forward(self, input, proto_prev_stage=True, prev_qry_fea=None, prev_sup_fea=None, cate_mapping=None):

		img, support, support_mask = input

		# Encoder
		img_features, _, _ = self.oneshot_highlvl_forward(img)
		sup_features, _, _ = self.oneshot_highlvl_forward(support)

		# Knowledge infusion module
		if prev_qry_fea is not None and prev_sup_fea is not None:
			if prev_qry_fea.shape != img_features.shape:
				prev_qry_fea = F.upsample(prev_qry_fea, size=img_features.size()[2:], mode='bilinear',
				                          align_corners=True)
				prev_sup_fea = F.upsample(prev_sup_fea, size=sup_features.size()[2:], mode='bilinear',
				                          align_corners=True)

			img_features = self.feature_fusion(torch.cat([prev_qry_fea, img_features], dim=1))
			sup_features = self.feature_fusion(torch.cat([prev_sup_fea, sup_features], dim=1))

		batch_n, _, mask_h, mask_w = sup_features.size()
		support_mask = F.upsample(support_mask, size=(mask_h, mask_w), mode='nearest')

		# Get indexes for each class
		maskmap = self.mask2map(support_mask.squeeze(1), self.cate_num)

		# Compute average features using the indexes
		sp_ave_features = torch.matmul(maskmap.view(batch_n, self.cate_num, mask_h * mask_w),
		                               # batch * class_num * hw
		                               sup_features.permute(0, 2, 3, 1).view(batch_n, mask_h * mask_w,
		                                                                     self.hidden_layers)) # batch * hw * feature channels
																				# batch * classnum * feature channels

		sup_classes = torch.unique(support_mask).long()

		if self.DML_mode == 'fixed':
			dml = self.dual_metric_fixed
		else:
			dml = self.dual_metric_ucs

		att_mask, sim_mask = dml(sp_ave_features, img_features, self.cate_num, proto_prev_stage, sup_classes)

		return F.upsample(att_mask, size=img.size()[2:], mode='bilinear', align_corners=True), F.upsample(
			self.scaler * sim_mask, size=img.size()[2:], mode='bilinear', align_corners=True)

	def dual_metric_ucs(self, sp_ave_features, features, nclasses, proto_prev_stage, sup_classes_bg):
		# This is the DML methods using unseen class screening (ucs),
		# where the background prototype is aggregated by
		# (sum human cls prototypes) / (# of human cls prototypes annotated in support image).

		# Get rid of the background index and remain the existing indexes
		# We only calculate
		max_classes = nclasses
		sup_classes = sup_classes_bg[sup_classes_bg != 0]
		screened_sp = sp_ave_features[:, sup_classes, :]

		# Features initialization: AGM (denoted as att) and NCM (denoted as sim)
		sem_lis = []
		sim_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		att_lis = []
		att_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		# To reduce memory usage, we use loops to process features for each batch in each class
		for i in range(len(sup_classes)):

			class_fea = screened_sp[:, i, :]

			# Generate class features and get similarity mask
			temp_list = []
			for b_ind in range(features.shape[0]):
				batch_fea = class_fea[b_ind, :]

				# When we don't want this class to be parsed, prototype for this class is set to 0
				if torch.sum(batch_fea) == 0:
					proto_fea = torch.zeros_like(batch_fea)

				else:
					if proto_prev_stage or torch.sum(self.prototype[i]) == 0:
						# When we want to parse this class but the prototype is 0, we use the batch_fea on the go
						proto_fea = batch_fea
					else:
						proto_num = sup_classes[i]
						if proto_prev_stage or torch.sum(self.prototype[proto_num]) == 0:
							# When we want to parse this class but the prototype is 0, we use batch_fea
							proto_fea = batch_fea
						else:
							# When batch_fea != 0, calculate proto_fea
							proto_fea = (1 - self.beta) * self.prototype[
								proto_num].clone() + self.beta * batch_fea

					# If training, update prototype
					if self.training:
						self.prototype[i] = proto_fea.detach()

				batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
				                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
				temp_list.append(batch_tmp_seg_map)

			# Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
			tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
			bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
			sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
			sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

			att_fea = self.decoder3(self.decoder2(tmp_seg_map.unsqueeze(1) * features + features))
			att_mask = self.classifier_6(att_fea)
			att_lis.append(att_mask[:, 0, :, :].unsqueeze(1))
			att_bg_fea += att_mask[:, 1, :, :].unsqueeze(1)

		if len(sup_classes) != 0:
			sim_bg_fea = sim_bg_fea / len(sup_classes)
			att_bg_fea = att_bg_fea / len(sup_classes)
		else:
			sim_bg_fea = torch.ones_like(sim_bg_fea).cuda()
			att_bg_fea = torch.ones_like(att_bg_fea).cuda()

		sem_lis = [self.bg_sim_fusion(sim_bg_fea)] + sem_lis
		att_lis = [self.bg_att_fusion(att_bg_fea)] + att_lis

		# bg has to be added
		sim_mask_screened = torch.cat(sem_lis, 1)
		att_mask_screened = torch.cat(att_lis, 1)

		sim_mask, att_mask = torch.zeros(features.shape[0], max_classes, features.shape[2], features.shape[3]).cuda(), \
		                     torch.zeros(features.shape[0], max_classes, features.shape[2], features.shape[3]).cuda()

		# Follow the previous indexes
		sim_mask[:, sup_classes_bg], att_mask[:, sup_classes_bg] = sim_mask_screened, att_mask_screened

		return att_mask, sim_mask

	def dual_metric_fixed(self, sp_ave_features, features, nclasses, proto_prev_stage, sup_classes_bg):

		# This is the DML methods not using unseen class screening (ucs),
		# where the background prototype is aggregated by (sum of human cls prototypes) / (a fixed term).
		# (Variable: sup_classes_bg) is not used in this function.

		# Features initialization: AGM (denoted as att) and NCM (denoted as sim)
		sem_lis = []
		sim_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		att_lis = []
		att_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		# To reduce memory usage, we use loops to process features for each batch in each class
		for i in range(1, nclasses):

			class_fea = sp_ave_features[:, i, :]

			# Generate class features and get similarity mask
			temp_list = []
			for b_ind in range(features.shape[0]):
				batch_fea = class_fea[b_ind, :]

				# When we don't want this class to be parsed, prototype for this class is set to 0
				if torch.sum(batch_fea) == 0:
					proto_fea = torch.zeros_like(batch_fea)

				else:
					if proto_prev_stage or torch.sum(self.prototype[i]) == 0:
						# When we want to parse this class but the prototype is 0, we use the batch_fea on the go
						proto_fea = batch_fea
					else:
						# If batch_fea != 0, we form dymanic proto_fea
						proto_fea = (1 - self.beta) * self.prototype[
							i].clone() + self.beta * batch_fea

					# If training, update prototype
					if self.training:
						self.prototype[i] = proto_fea.detach()

				batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
				                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
				temp_list.append(batch_tmp_seg_map)

			# Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
			tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
			bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
			sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
			sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

			att_fea = self.decoder3(self.decoder2(tmp_seg_map.unsqueeze(1) * features + features))
			att_mask = self.classifier_6(att_fea)
			att_lis.append(att_mask[:, 0, :, :].unsqueeze(1))
			att_bg_fea += att_mask[:, 1, :, :].unsqueeze(1)

		sim_bg_fea = sim_bg_fea / (nclasses - 1)
		att_bg_fea = att_bg_fea / (nclasses - 1)

		sem_lis = [self.bg_sim_fusion(sim_bg_fea)] + sem_lis
		att_lis = [self.bg_att_fusion(att_bg_fea)] + att_lis

		sim_mask = torch.cat(sem_lis, 1)
		att_mask = torch.cat(att_lis, 1)

		return att_mask, sim_mask


# Test whether batch-wise same support works
class popnet_kway_dp_samesup(
	deeplab_xception_synBN.DeepLabv3_plus_v2):
	def __init__(self, nInputChannels=3, n_classes=7, os=16, hidden_layers=256, beta=0.001, scaler=10.,
	             feature_lvl='high', DML_mode='fixed'):
		super(popnet_kway_dp_samesup, self).__init__(nInputChannels=nInputChannels,
		                                             n_classes=n_classes,
		                                             os=os, pretrained=True)

		# Settings
		self.hidden_layers = hidden_layers
		self.DML_mode = DML_mode
		self.cos_similarity_func = nn.CosineSimilarity()
		self.feature_lvl = feature_lvl
		self.beta = beta

		# AGM layers
		self.classifier_6 = nn.Conv2d(256, 2, kernel_size=1)
		self.bg_att_fusion = nn.Conv2d(1, 1, kernel_size=1)

		# NCM layers
		self.after_sim_fg = nn.Conv2d(1, 1, kernel_size=1)
		self.after_sim_bg = nn.Conv2d(1, 1, kernel_size=1)
		self.bg_sim_fusion = nn.Conv2d(1, 1, kernel_size=1)
		self.scaler = nn.Parameter(torch.tensor(scaler), requires_grad=True)  # Learnable scaler

		# KIM layers
		self.feature_fusion = nn.Sequential(
			deeplab_xception_synBN.Decoder_module(512, 256),
			deeplab_xception_synBN.Decoder_module(256, 256),
		)

		self.prototype = torch.nn.Parameter(torch.zeros(17, 256), requires_grad=False)

	def mask2map(self, mask, class_num):
		# Helper function for getting feature indexes for each class (gpu)

		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w).cuda()

		for i in range(class_num):
			class_pix = torch.where(mask == i, torch.ones(1).cuda(), torch.zeros(1).cuda())
			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			class_sum = torch.where(class_sum == 0, torch.ones(1).cuda(), class_sum)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			maskmap_ave[:, i, :, :] = class_pix_ave

		return maskmap_ave

	def mask2map_cpu(self, mask, class_num):
		# Helper function for getting feature indexes for each class
		n, h, w = mask.shape

		maskmap_ave = torch.zeros(n, class_num, h, w)

		for i in range(class_num):
			class_pix = torch.where(mask == i, torch.ones(1), torch.zeros(1))
			class_sum = torch.sum(class_pix.view(n, h * w), dim=1)
			class_sum = torch.where(class_sum == 0, torch.ones(1), class_sum)
			class_pix_ave = class_pix / class_sum.view(n, 1, 1)

			maskmap_ave[:, i, :, :] = class_pix_ave

		return maskmap_ave

	def forward(self, input, cate_num=17, proto_prev_stage=True, prev_qry_fea=None, prev_sup_fea=None):

		nclasses = cate_num
		img, support_difsup, support_mask_difsup = input

		support, support_mask = support_difsup[0], support_mask_difsup[0]
		support, support_mask = support.expand(support_difsup.shape), support_mask.expand(support_mask_difsup.shape)

		# print(support.shape, support_mask.shape)
		# Encoder
		img_features = self.oneshot_flex_forward(img, feature_lvl=self.feature_lvl)
		sup_features = self.oneshot_flex_forward(support, feature_lvl=self.feature_lvl)

		# Knowledge infusion module
		if prev_qry_fea is not None and prev_sup_fea is not None:
			if prev_qry_fea.shape != img_features.shape:
				prev_qry_fea = F.upsample(prev_qry_fea, size=img_features.size()[2:], mode='bilinear',
				                          align_corners=True)
				prev_sup_fea = F.upsample(prev_sup_fea, size=sup_features.size()[2:], mode='bilinear',
				                          align_corners=True)

			img_features = self.feature_fusion(torch.cat([prev_qry_fea, img_features], dim=1))
			sup_features = self.feature_fusion(torch.cat([prev_sup_fea, sup_features], dim=1))

		batch_n, _, mask_h, mask_w = sup_features.size()
		support_mask = F.upsample(support_mask, size=(mask_h, mask_w), mode='nearest')

		# Get indexes for each class
		maskmap = self.mask2map(support_mask.squeeze(1), nclasses)

		# Compute average features using the indexes
		sp_ave_features = torch.matmul(maskmap.view(batch_n, nclasses, mask_h * mask_w),
		                               # batch * class_num * hw
		                               sup_features.permute(0, 2, 3, 1).view(batch_n, mask_h * mask_w,
		                                                                     self.hidden_layers)
		                               # batch * hw * feature channels
		                               )  # batch * classnum * feature channels

		sup_classes = torch.unique(support_mask).long()

		if self.DML_mode == 'fixed':
			dml = self.dual_metric_fixed
		else:
			dml = self.dual_metric_ucs

		att_mask, sim_mask = dml(sp_ave_features, img_features, nclasses, proto_prev_stage, sup_classes)

		return F.upsample(att_mask, size=img.size()[2:], mode='bilinear', align_corners=True), F.upsample(
			self.scaler * sim_mask, size=img.size()[2:], mode='bilinear', align_corners=True)

	def dual_metric_ucs(self, sp_ave_features, features, nclasses, proto_prev_stage, sup_classes_bg):
		# This is the DML methods using unseen class screening (ucs),
		# where the background prototype is aggregated by
		# (sum human cls prototypes) / (# of human cls prototypes annotated in support image).

		# Get rid of the background index and remain the existing indexes
		# We only calculate
		max_classes = nclasses
		sup_classes = sup_classes_bg[sup_classes_bg != 0]
		screened_sp = sp_ave_features[:, sup_classes, :]

		# Features initialization: AGM (denoted as att) and NCM (denoted as sim)
		sem_lis = []
		sim_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		att_lis = []
		att_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		# To reduce memory usage, we use loops to process features for each batch in each class
		for i in range(len(sup_classes)):

			class_fea = screened_sp[:, i, :]

			# Generate class features and get similarity mask
			temp_list = []
			for b_ind in range(features.shape[0]):
				batch_fea = class_fea[b_ind, :]

				# When we don't want this class to be parsed, prototype for this class is set to 0
				if torch.sum(batch_fea) == 0:
					proto_fea = torch.zeros_like(batch_fea)

				else:
					if proto_prev_stage or torch.sum(self.prototype[i]) == 0:
						# When we want to parse this class but the prototype is 0, we use the batch_fea on the go
						proto_fea = batch_fea
					else:
						proto_num = sup_classes[i]
						if proto_prev_stage or torch.sum(self.prototype[proto_num]) == 0:
							# When we want to parse this class but the prototype is 0, we use batch_fea
							proto_fea = batch_fea
						else:
							# When batch_fea != 0, calculate proto_fea
							proto_fea = (1 - self.beta) * self.prototype[
								proto_num].clone() + self.beta * batch_fea

					# If training, update prototype
					if self.training:
						self.prototype[i] = proto_fea.detach()

				batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
				                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
				temp_list.append(batch_tmp_seg_map)

			# Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
			tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
			bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
			sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
			sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

			att_fea = self.decoder3(self.decoder2(tmp_seg_map.unsqueeze(1) * features + features))
			att_mask = self.classifier_6(att_fea)
			att_lis.append(att_mask[:, 0, :, :].unsqueeze(1))
			att_bg_fea += att_mask[:, 1, :, :].unsqueeze(1)

		if len(sup_classes) != 0:
			sim_bg_fea = sim_bg_fea / len(sup_classes)
			att_bg_fea = att_bg_fea / len(sup_classes)
		else:
			sim_bg_fea = torch.ones_like(sim_bg_fea).cuda()
			att_bg_fea = torch.ones_like(att_bg_fea).cuda()

		sem_lis = [self.bg_sim_fusion(sim_bg_fea)] + sem_lis
		att_lis = [self.bg_att_fusion(att_bg_fea)] + att_lis

		# bg has to be added
		sim_mask_screened = torch.cat(sem_lis, 1)
		att_mask_screened = torch.cat(att_lis, 1)

		sim_mask, att_mask = torch.zeros(features.shape[0], max_classes, features.shape[2], features.shape[3]).cuda(), \
		                     torch.zeros(features.shape[0], max_classes, features.shape[2], features.shape[3]).cuda()

		# Follow the previous indexes
		sim_mask[:, sup_classes_bg], att_mask[:, sup_classes_bg] = sim_mask_screened, att_mask_screened

		return att_mask, sim_mask

	def dual_metric_fixed(self, sp_ave_features, features, nclasses, proto_prev_stage, sup_classes_bg):

		# This is the DML methods not using unseen class screening (ucs),
		# where the background prototype is aggregated by (sum of human cls prototypes) / (a fixed term).
		# (Variable: sup_classes_bg) is not used in this function.

		# Features initialization: AGM (denoted as att) and NCM (denoted as sim)
		sem_lis = []
		sim_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		att_lis = []
		att_bg_fea = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3]).cuda()

		# To reduce memory usage, we use loops to process features for each batch in each class
		for i in range(1, nclasses):

			class_fea = sp_ave_features[:, i, :]

			# Generate class features and get similarity mask
			temp_list = []
			for b_ind in range(features.shape[0]):
				batch_fea = class_fea[b_ind, :]

				# When we don't want this class to be parsed, prototype for this class is set to 0
				if torch.sum(batch_fea) == 0:
					proto_fea = torch.zeros_like(batch_fea)

				else:
					if proto_prev_stage or torch.sum(self.prototype[i]) == 0:
						# When we want to parse this class but the prototype is 0, we use the batch_fea on the go
						proto_fea = batch_fea
					else:
						# If batch_fea != 0, we form dymanic proto_fea
						proto_fea = (1 - self.beta) * self.prototype[
							i].clone() + self.beta * batch_fea

					# If training, update prototype
					if self.training:
						self.prototype[i] = proto_fea.detach()

				batch_tmp_seg_map = self.cos_similarity_func(features[b_ind, :, :, :].unsqueeze(0),
				                                             proto_fea.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))
				temp_list.append(batch_tmp_seg_map)

			# Aggregate features for AGM (denoted as att) and NCM (denoted as sim)
			tmp_seg_map = torch.stack(temp_list, dim=0).squeeze(1)
			bg_tmp_seg_map = torch.ones_like(tmp_seg_map.unsqueeze(1)).cuda() - tmp_seg_map.unsqueeze(1)
			sem_lis.append(self.after_sim_fg(tmp_seg_map.unsqueeze(1)))
			sim_bg_fea += self.after_sim_bg(bg_tmp_seg_map)

			att_fea = self.decoder3(self.decoder2(tmp_seg_map.unsqueeze(1) * features + features))
			att_mask = self.classifier_6(att_fea)
			att_lis.append(att_mask[:, 0, :, :].unsqueeze(1))
			att_bg_fea += att_mask[:, 1, :, :].unsqueeze(1)

		sim_bg_fea = sim_bg_fea / (nclasses - 1)
		att_bg_fea = att_bg_fea / (nclasses - 1)

		sem_lis = [self.bg_sim_fusion(sim_bg_fea)] + sem_lis
		att_lis = [self.bg_att_fusion(att_bg_fea)] + att_lis

		sim_mask = torch.cat(sem_lis, 1)
		att_mask = torch.cat(att_lis, 1)

		return att_mask, sim_mask


class popnet_1way(
	deeplab_xception_synBN.DeepLabv3_plus_v2):
	def __init__(self, nInputChannels=3, n_classes=7, os=16, hidden_layers=256, alpha=20):
		super(popnet_1way, self).__init__(nInputChannels=nInputChannels,
		                                  n_classes=n_classes,
		                                  os=os, pretrained=True)

		# Settings
		self.hidden_layers = hidden_layers
		self.cos_similarity_func = nn.CosineSimilarity()

		# AGM layers
		self.classifier_6 = nn.Sequential(
			nn.Conv2d(256, 2, kernel_size=1),
		)

		# NCM layers
		self.after_sim = nn.Sequential(
			nn.Conv2d(1, 2, kernel_size=1),
		)

		# KIM layers
		self.feature_fusion = nn.Sequential(
			deeplab_xception_synBN.Decoder_module(512, 256),
			deeplab_xception_synBN.Decoder_module(256, 256),
		)

	def cateogrory_feature(self, features, mask):
		# Helper function for getting feature indexes for the only class
		batch_n, _, mask_w, mask_h = features.size()
		pos_sum = torch.sum(
			mask.view(batch_n, mask_h * mask_w), dim=1).unsqueeze(1)
		pos_sum = torch.where(pos_sum == 0, torch.ones(1).cuda(), pos_sum)
		vec_pos = torch.sum(torch.sum(features * mask, dim=3), dim=2) / pos_sum

		return vec_pos

	def forward(self, input, cate_num=None, prev_qry_fea=None, prev_sup_fea=None):
		img, support, sup_known = input

		# Encoder
		img_features = self.oneshot_forward(img)
		sup_features = self.oneshot_forward(support)

		# Knowledge infusion module
		img_features = self.feature_fusion(torch.cat([prev_qry_fea, img_features], dim=1))
		sup_features = self.feature_fusion(torch.cat([prev_sup_fea, sup_features], dim=1))

		batch_n, _, mask_w, mask_h = sup_features.size()
		sup_known = F.upsample(sup_known, size=(mask_w, mask_h), mode='nearest')

		# Get indexes for the class
		sup_known_features = self.cateogrory_feature(sup_features, sup_known)

		# DML
		tmp_seg = self.cos_similarity_func(img_features, sup_known_features.unsqueeze(dim=2).unsqueeze(dim=3))
		tmp_seg = tmp_seg.unsqueeze(dim=1)
		res_features = img_features + img_features * tmp_seg
		res_features = self.decoder3(self.decoder2(res_features))
		sim_mask = tmp_seg
		sim_mask = self.after_sim(sim_mask)
		img_mask = self.classifier_6(res_features)

		return F.upsample(img_mask, size=img.size()[2:], mode='bilinear', align_corners=True), \
		       F.upsample(sim_mask, size=img.size()[2:], mode='bilinear', align_corners=True)

	def prev_forward(self, input, prev_qry_fea=None, prev_sup_fea=None):
		# prepare parent qry_features and parent sup_features for the next stage
		img, support = input

		img_features = self.oneshot_forward(img)
		sup_features = self.oneshot_forward(support)

		img_features = self.feature_fusion(torch.cat([prev_qry_fea, img_features], dim=1))
		sup_features = self.feature_fusion(torch.cat([prev_sup_fea, sup_features], dim=1))

		return img_features, sup_features


class popnet_1way_proto(
	deeplab_xception_synBN.DeepLabv3_plus_v2):
	def __init__(self, nInputChannels=3, n_classes=7, os=16, hidden_layers=256, alpha=20, beta=0.001, scaler=10., cate_num=17):
		super(popnet_1way_proto, self).__init__(nInputChannels=nInputChannels,
		                                        n_classes=n_classes,
		                                        os=os, pretrained=False)

		# Settings
		self.hidden_layers = hidden_layers
		self.cos_similarity_func = nn.CosineSimilarity()
		self.scaler = nn.Parameter(torch.tensor(scaler), requires_grad=True)
		self.beta = beta

		# AGM layers
		self.classifier_6 = nn.Sequential(
			nn.Conv2d(256, 2, kernel_size=1),
		)

		# NCM layers
		self.after_sim = nn.Sequential(
			nn.Conv2d(1, 2, kernel_size=1),
		)

		# KIM layers
		self.feature_fusion = nn.Sequential(
			deeplab_xception_synBN.Decoder_module(512, 256),
			deeplab_xception_synBN.Decoder_module(256, 256),
		)

		self.prototype = torch.nn.Parameter(torch.zeros(cate_num, 256), requires_grad=False)

	def cateogrory_feature(self, features, mask):
		batch_n, _, mask_w, mask_h = features.size()
		pos_sum = torch.sum(
			mask.view(batch_n, mask_h * mask_w), dim=1).unsqueeze(1)
		pos_sum = torch.where(pos_sum == 0, torch.ones(1).cuda(), pos_sum)

		# temp_sum = torch.sum(torch.sum(features * mask, dim=3), dim=2)
		# print(torch.sum(torch.sum(features * mask, dim=3), dim=2).shape, pos_sum.shape)
		vec_pos = torch.sum(torch.sum(features * mask, dim=3), dim=2) / pos_sum

		# print(torch.sum(temp_sum))
		# assert not (temp_sum != temp_sum).any()
		# assert not (vec_pos != vec_pos).any()

		return vec_pos

	def forward(self, input, proto_prev_stage=True, prev_qry_fea=None, prev_sup_fea=None, cate_mapping=None):

		# In 1way OSHP, all the foreground classes are annotated as 1,
		# hence we track a mapping: (category_index -> prototype_index).
		# cate_mapping = [index_for_batch1, index_for_batch2]
		assert cate_mapping is not None
		img, support, support_mask = input

		img_high, _, _ = self.oneshot_highlvl_forward(img)
		sup_high, _, _ = self.oneshot_highlvl_forward(support)

		img_high = self.feature_fusion(torch.cat([prev_qry_fea, img_high], dim=1))
		sup_high = self.feature_fusion(torch.cat([prev_sup_fea, sup_high], dim=1))

		batch_n, _, mask_h, mask_w = sup_high.size()
		support_mask = F.upsample(support_mask, size=(mask_h, mask_w), mode='nearest')

		sp_ave_features = self.cateogrory_feature(sup_high, support_mask)

		# sem_lis = [bg, fg1, fg2, fg3...]
		att_mask, sim_mask = self.dual_metric(sp_ave_features, img_high, proto_prev_stage, cate_mapping)

		return F.upsample(att_mask, size=img.size()[2:], mode='bilinear', align_corners=True), F.upsample(
			self.scaler * sim_mask, size=img.size()[2:], mode='bilinear', align_corners=True)

	def prev_forward(self, input, prev_qry_fea=None, prev_sup_fea=None):

		# prepare parent qry_features and parent sup_features for the next stage
		img, support = input

		img_features = self.oneshot_forward(img)
		sup_features = self.oneshot_forward(support)

		img_features = self.feature_fusion(torch.cat([prev_qry_fea, img_features], dim=1))
		sup_features = self.feature_fusion(torch.cat([prev_sup_fea, sup_features], dim=1))

		return img_features, sup_features

	def dual_metric(self, sp_ave_features, features, proto_prev_stage, cate_mapping):
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
				if proto_prev_stage or torch.sum(self.prototype[i]) == 0:
					# When we want to parse this class but the prototype is 0, we use the batch_fea on the go
					proto_fea = batch_fea
				else:
					# If batch_fea != 0, we form dymanic proto_fea
					proto_fea = (1 - self.beta) * self.prototype[
						i].clone() + self.beta * batch_fea

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
		sim_mask = tmp_seg
		sim_mask = self.after_sim(sim_mask)
		att_mask = self.classifier_6(res_features)

		return att_mask, sim_mask
