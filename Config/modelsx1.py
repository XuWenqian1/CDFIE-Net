import torch.nn as nn
import torch
import torch.fft as fft
import torch.nn.functional as F

class CDFIENet(nn.Module):
    def __init__(self):
        super(CDFIENet, self).__init__()
        self.layer_0_r = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.layer_0_g = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)
        self.layer_0_b = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=1)

        self.layer_1_r = DegradationAwareResidualModule(9)
        self.layer_1_g = DegradationAwareResidualModule(9)
        self.layer_1_b = DegradationAwareResidualModule(9)

        self.layer_2_r = SpatialFrequencyEnhancementModule(9)
        self.layer_2_g = SpatialFrequencyEnhancementModule(9)
        self.layer_2_b = SpatialFrequencyEnhancementModule(9)

        self.layer_3 = MultiBranchExtractionModule(27, 9)
        self.layer_4 = InteractiveFusionAttentionModule(36)

        self.layer_tail = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, stride=1, padding=1, groups=36, bias=False),
            # 深度卷积
            nn.Conv2d(in_channels=36, out_channels=18, kernel_size=1, stride=1, padding=0, bias=False),  # 点卷积
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=18, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self,input):
        input_r = torch.unsqueeze(input[:,0,:,:], dim=1)
        input_g  = torch.unsqueeze(input[:,1,:,:], dim=1)
        input_b = torch.unsqueeze(input[:,2,:,:], dim=1)

        layer_0_r = self.layer_0_r(input_r)
        layer_0_g = self.layer_0_g(input_g)
        layer_0_b = self.layer_0_b(input_b)

        layer_1_r = self.layer_1_r(layer_0_r)
        layer_1_g = self.layer_1_g(layer_0_g)
        layer_1_b = self.layer_1_b(layer_0_b)

        layer_2_r = self.layer_2_r(layer_1_r)
        layer_2_g = self.layer_2_g(layer_1_g)
        layer_2_b = self.layer_2_b(layer_1_b)

        layer_concat = torch.cat([layer_2_r,layer_2_g,layer_2_b],dim=1)

        layer_3 = self.layer_3(layer_concat)
        layer_4 = self.layer_4(layer_3)

        output = self.layer_tail(layer_4)
        return output
        

