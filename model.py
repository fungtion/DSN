import torch.nn as nn
import torch
from functions import ReverseLayerF
from torch.autograd import Variable


class DSN(nn.Module):
    def __init__(self, code_size=100, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder_conv = nn.Sequential()
        self.source_encoder_conv.add_module('conv_pse1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,
                                                                padding=3))
        self.source_encoder_conv.add_module('ac_pse1', nn.ReLU(True))
        self.source_encoder_conv.add_module('bn_pse1', nn.BatchNorm2d(num_features=32))
        self.source_encoder_conv.add_module('pool_pse1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.source_encoder_conv.add_module('conv_pse2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                                padding=3))
        self.source_encoder_conv.add_module('ac_pse2', nn.ReLU(True))
        self.source_encoder_conv.add_module('bn_pse2', nn.BatchNorm2d(num_features=64))
        self.source_encoder_conv.add_module('pool_pse2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.source_encoder_fc = nn.Sequential()
        self.source_encoder_fc.add_module('fc_pse3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.source_encoder_fc.add_module('ac_pse3', nn.ReLU(True))
        self.source_encoder_fc.add_module('bn_pse3', nn.BatchNorm2d(code_size))

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential()
        self.target_encoder_conv.add_module('conv_pte1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,
                                                                padding=3))
        self.target_encoder_conv.add_module('ac_pte1', nn.ReLU(True))
        self.target_encoder_conv.add_module('bn_pte1', nn.BatchNorm2d(num_features=32))
        self.target_encoder_conv.add_module('pool_pte1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.target_encoder_conv.add_module('conv_pte2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                                                                padding=3))
        self.target_encoder_conv.add_module('ac_pte2', nn.ReLU(True))
        self.target_encoder_conv.add_module('bn_pte2', nn.BatchNorm2d(num_features=64))
        self.target_encoder_conv.add_module('pool_pte2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.target_encoder_fc = nn.Sequential()
        self.target_encoder_fc.add_module('fc_pte3', nn.Linear(in_features=7 * 7 * 64, out_features=code_size))
        self.target_encoder_fc.add_module('ac_pte3', nn.ReLU(True))
        self.target_encoder_fc.add_module('bn_pte3', nn.BatchNorm2d(num_features=code_size))

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.shared_encoder_conv = nn.Sequential()
        self.shared_encoder_conv.add_module('conv_se1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                  padding=3))
        self.shared_encoder_conv.add_module('ac_sa1', nn.ReLU(True))
        self.shared_encoder_conv.add_module('pool_se1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.shared_encoder_conv.add_module('conv_se2', nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,
                                                                  padding=3))
        self.shared_encoder_conv.add_module('ac_se2', nn.ReLU(True))
        self.shared_encoder_conv.add_module('pool_se2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.shared_encoder_fc = nn.Sequential()
        self.shared_encoder_fc.add_module('fc_se3', nn.Linear(in_features=7 * 7 * 48, out_features=100))
        self.shared_encoder_fc.add_module('ac_se3', nn.ReLU(True))
        self.shared_encoder_fc.add_module('fc_se4', nn.Linear(in_features=100, out_features=code_size))
        self.shared_encoder_fc.add_module('ac_se4', nn.ReLU(True))

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Linear(in_features=100, out_features=n_class)

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se5', nn.Linear(in_features=100, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se5', nn.ReLU(True))
        self.shared_encoder_pred_domain.add_module('drop_se5', nn.Dropout2d())

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=100, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=2 * code_size, out_features=300))
        self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))
        self.shared_decoder_fc.add_module('bn_sd1', nn.BatchNorm2d(num_features=300))

        self.shared_decoder_conv = nn.Sequential()
        self.shared_decoder_conv.add_module('conv_sd2', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd2', nn.ReLU())
        self.shared_decoder_conv.add_module('bn_sd2', nn.BatchNorm2d(num_features=16))

        self.shared_decoder_conv.add_module('conv_sd3', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd3', nn.ReLU())
        self.shared_decoder_conv.add_module('bn_sd3', nn.BatchNorm2d(num_features=16))

        self.shared_decoder_conv.add_module('us_sd4', nn.Upsample(size=28, mode='bilinear'))

        self.shared_decoder_conv.add_module('conv_sd5', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd5', nn.ReLU(True))
        self.shared_decoder_conv.add_module('bn_sd5', nn.BatchNorm2d(16))

        self.shared_decoder_conv.add_module('conv_sd6', nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('bn_sd6', nn.BatchNorm2d(num_features=1))

    def forward(self, source_data, target_data, alpha):

        # source private encoder
        source_private_feat = self.source_encoder_conv(source_data)
        source_private_feat = source_private_feat.view(-1, 64 * 7 * 7)
        source_private_code = self.source_encoder_fc(source_private_feat)

        # target private encoder
        target_private_feat = self.target_encoder_conv(target_data)
        target_private_feat = target_private_feat.view(-1, 64 * 7 * 7)
        target_private_code = self.target_encoder_fc(target_private_feat)

        # source shared encoder
        source_shared_feat = self.shared_encoder_conv(source_data)
        source_shared_feat = source_shared_feat.view(-1, 48 * 7 * 7)
        source_shared_code = self.shared_encoder_fc(source_shared_feat)

        source_class_label = self.shared_encoder_pred_class(source_shared_code)

        reversed_source_shared_code = ReverseLayerF.apply(source_shared_code, alpha)
        source_domain_label = self.shared_encoder_pred_domain(reversed_source_shared_code)

        # target shared encoder
        target_shared_feat = self.shared_encoder_conv(target_data)
        target_shared_feat = target_shared_feat.view(-1, 48 * 7 * 7)
        target_shared_code = self.shared_encoder_fc(target_shared_feat)

        reversed_target_shared_code = ReverseLayerF.apply(target_shared_code, alpha)
        target_domain_label = self.shared_encoder_pred_domain(reversed_target_shared_code)

        # source shared decoder
        source_code = torch.cat((source_private_code, source_shared_code), 1)
        rec_source_vec = self.shared_decoder_fc(source_code)
        rec_source_vec = rec_source_vec.view(-1, 3, 10, 10)

        rec_source = self.shared_decoder_conv(rec_source_vec)

        # target shared decoder
        target_code = torch.cat((target_private_code, target_shared_code), 1)
        rec_target_vec = self.shared_decoder_fc(target_code)
        rec_target_vec = rec_target_vec.view(-1, 3, 10, 10)
        rec_target = self.shared_decoder_conv(rec_target_vec)

        # test reconstruction for target shared code only
        target_shared_code_test = torch.cat((Variable(torch.zeros(self.code_size)), target_shared_code), 1)
        rec_target_shared = self.shared_decoder_fc(target_shared_code_test)
        rec_target_shared = rec_target_shared.view(- 1, 3, 10, 10)
        rec_target_shared = self.shared_decoder_conv(rec_target_shared)

        # test reconstruction for target private code only
        target_private_code_test = torch.cat((target_private_code, Variable(torch.zeros(self.code_size))), 1)
        rec_target_private = self.shared_decoder_fc(target_private_code_test)
        rec_target_private = rec_target_private.view(-1, 3, 10, 10)
        rec_target_private = self.shared_decoder_conv(rec_target_private)

        return source_private_code, target_private_code, source_shared_code, target_shared_code, \
               source_class_label, source_domain_label, target_domain_label, rec_source, rec_target, \
               rec_target_private, rec_target_shared





