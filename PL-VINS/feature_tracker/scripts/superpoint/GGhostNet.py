import torch
import torch.nn as nn
from pathlib import Path

from .modules.g_ghost_backbone import GGhost_Backbone
from .modules.cnn_heads import DetectorHead, DescriptorHead

class SuperPoint_GGhost(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, args):
        super(SuperPoint_GGhost, self).__init__()
        print(" Running SuperPoint:",args.model_name)
        self.backbone = GGhost_Backbone(args.gghost)
        self.detector_head = DetectorHead(input_channel=128,
                                          grid_size=8, using_bn=True)
        self.descriptor_head = DescriptorHead(input_channel=128,
                                              output_channel=256,
                                              grid_size=8, using_bn=True)
        model_path=args.gghost032_model if args.gghost=='032' else args.gghost080_model
        if  args.cuda:
            # Train on GPU, deploy on GPU.
            checkpoint = torch.load(model_path)
        else:
            # Train on GPU, deploy on CPU.
            checkpoint = torch.load(model_path,map_location=torch.device('cpu'))        
        self.load_state_dict(checkpoint)#）

    def forward(self, x):
        feat_map = self.backbone(x)
        scores = self.detector_head(feat_map)
        descriptors = self.descriptor_head(feat_map)
        return scores,descriptors


if __name__=='__main__':
    from fvcore.nn import FlopCountAnalysis,flop_count_str,flop_count_table
    import yaml
    config_file = '/home/xiangyunfei/nfs/image-matching-toolbox/configs/superpoint.yml'
    with open(config_file, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)

    flops = FlopCountAnalysis(SuperPoint_GGhost(model_conf['default']),torch.randn((1, 1, 480,720) ))
    # Results
    print(flops.total())
    # print(flop_count_str(flops))
    print(flop_count_table(flops,max_depth=2))
    #model.load_state_dict(torch.load('../superpoint_bn.pth'))
    # print('Done')
