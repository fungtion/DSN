import torch.nn as nn
import torch.cat
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

        ######################################
        # TODO: shared cdcoder (large decoder)
        ######################################

    def forward(self, input_data, mode, rec_scheme, alpha):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':

            # target private encoder
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 64 * 7 * 7)
            private_code = self.target_encoder_fc(private_feat)

        result.append(private_code)

        # shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 48 * 7 * 7)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, alpha)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            concat_code = torch.cat((Variable(torch.zeros(self.code_size)), shared_code), 1)
        elif rec_scheme == 'all':
            concat_code = torch.cat((private_code, shared_code), 1)

        rec_vec = self.shared_decoder_fc(concat_code)
        rec_vec = rec_vec.view(-1, 3, 10, 10)

        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result





