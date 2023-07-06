# 网络配置：080 G_GhostNet 前两层
# 仅用来训练: avgpool动态池化，以及不手动UpSample,以及替换SliceCONV,改成slice，以及line 143行downsample操作仍然保留
# 后续想优化SliceCONV，可以问问群里的iCe的老哥怎么做图优化

# backbone fixed training:
# 先读取，修改，再存，
# 尝试fix-1个epoch之后，解fix训练一下




import torch
import torch.nn as nn
import torch.nn.functional as F

# G_Ghost_RegNet
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes * self.expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, width // min(width, group_width), dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
       

class Stage(nn.Module):
    def __init__(self, block, inplanes, planes, group_width, blocks, stride=1, dilate=False, cheap_ratio=0.5):
        super(Stage, self).__init__()
        norm_layer = nn.BatchNorm2d
        downsample = None
        self.dilation = 1
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
            
        self.base = block(inplanes, planes, stride, downsample, group_width,
                            previous_dilation, norm_layer)
        self.end = block(planes, planes, group_width=group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer)
        
        group_width = int(group_width * 0.75)
        raw_planes = int(planes * (1 - cheap_ratio) / group_width) * group_width
        cheap_planes = planes - raw_planes
        self.cheap_planes = cheap_planes
        self.raw_planes = raw_planes
        

        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes+raw_planes*(blocks-2), cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_planes, cheap_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(cheap_planes, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap_relu = nn.ReLU(inplace=True)
        
        layers = []
        downsample = nn.Sequential(
            LambdaLayer(lambda x: x[:, :raw_planes])
        )
        # downsample=None

        layers = []
        layers.append(block(raw_planes, raw_planes, 1, downsample, group_width,
                            self.dilation, norm_layer))
        inplanes = raw_planes
        for _ in range(2, blocks-1):
            layers.append(block(inplanes, raw_planes, group_width=group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)
       
        # My Slice Operations
        # print(planes,self.raw_planes)
        # self.slice_conv1=SliceByConv(planes,0,self.raw_planes) 
        # self.slice_conv2=SliceByConv(planes,self.raw_planes,planes)
        # self.upsample1= nn.Upsample(scale_factor=(6,9))
        # self.upsample2= nn.Upsample(scale_factor=(5,5))      
        
    
    def forward(self, input):
        x0 = self.base(input)
        # print(x0.shape,self.raw_planes)
        m_list = [x0]
        e = x0[:, :self.raw_planes]   
        # e=self.slice_conv1(x0)    
        for l in self.layers:
            e = l(e)
            m_list.append(e)
        m = torch.cat(m_list,1)
        # print(m.shape)
        m = self.merge(m)
        
        # m=self.upsample1(m)
        # m=self.upsample2(m)
        
        c = x0[:, self.raw_planes:]
        # c=self.slice_conv2(x0) 
        # print("upsample shape:",c.shape)  
        c = self.cheap_relu(self.cheap(c)+m)
        
        x = torch.cat((e,c),1)
        x = self.end(x)
        return x


class GGhostRegNet(nn.Module):

    def __init__(self, block, layers, widths, num_classes=1000, zero_init_residual=True,
                 group_width=1, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(GGhostRegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.group_width = group_width
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        
        self.inplanes = widths[0]
        if layers[1] > 2:
            self.layer2 = Stage(block, self.inplanes, widths[1], group_width, layers[1], stride=2,
                          dilate=replace_stride_with_dilation[1], cheap_ratio=0.5) 
        else:      
            self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        
        self.inplanes = widths[1]
        self.layer3 = Stage(block, self.inplanes, widths[2], group_width, layers[2], stride=2,
                      dilate=replace_stride_with_dilation[2], cheap_ratio=0.5)
        
        self.inplanes = widths[2]
        if layers[3] > 2:
            self.layer4 = Stage(block, self.inplanes, widths[3], group_width, layers[3], stride=2,
                          dilate=replace_stride_with_dilation[3], cheap_ratio=0.5)
        else:
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[3])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.group_width,
                            previous_dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # return self._forward_impl(x)
        # 做了修改
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("layer_pre_out,",x.shape)
        x = self.layer1(x)
        # print("layer_1_out,",x.shape)
        x = self.layer2(x)
        # print("layer_2_out,",x.shape)
        # x = self.layer3(x)
        # x = self.layer4(x)
        return x




   
class GGhost_Backbone(torch.nn.Module):
    def __init__(self,type:str):
        super(GGhost_Backbone, self).__init__()  
        self.preblock=nn.Sequential(
            nn.Conv2d(1, 3, 3, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        if type=='032':
            self.module=GGhostRegNet(Bottleneck,   [2, 6, 15, 2], [96, 192, 432, 1008], group_width=48 )
        elif type=='080':
            self.module=GGhostRegNet(Bottleneck,  [2, 5, 15, 1], [80, 240, 720, 1920], group_width=120 )
        else:
            raise RuntimeError
        self.postblock=nn.Sequential(
            nn.Conv2d(240 if type=='080' else 192 , 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.upsample=nn.Upsample(scale_factor=(2,2)) #16/H,16/W -> 1/8H,1/8W 符合SP原版backbone输出大小
    def forward(self,x):
        x=self.preblock(x)
        x = self.module(x)
        x = self.upsample(x)
        x = self.postblock(x)
        return x


if __name__ == '__main__':
    
    checkpoint = torch.load('/share_data/g_ghost_regnet_8.0g_79.0.pth')
    model=GGhost_Backbone()
    for k in model.state_dict().keys():
        if k not in checkpoint.keys():
            print(k,"!!!!!")

    # print(list(model.state_dict().keys()))
    # print(list(checkpoint.keys()))
    model.load_state_dict(checkpoint,strict=False)# 仅保留同名参数
    torch.save(model.state_dict(),"test.pth")