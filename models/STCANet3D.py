import torch
import math
from torch import nn
from torch.nn import functional as F
from models.STAM import STIAUModule

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class SpatialAttn(nn.Module):
    def __init__(self, in_channels, number):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBlock(in_channels, number, 1)

    def forward(self, x):
        x = self.conv(x)
        a = torch.sigmoid(x)
        return a

class STCANet3D(nn.Module):
    def __init__(self, num_classes, use_gpu, loss={'xent', 'htri'}):
        super(STCANet3D, self).__init__()
        self.loss = loss
        self.use_gpu = use_gpu
        
        # 🌟 dynamic resnets1 load handle
        from models.resnets1 import resnet50_s1
        from models import inflate
        
        resnet2d = resnet50_s1(pretrained=True)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4)

        self.STCABlock3D2 = STCABlock3D(512)
        self.feat_dim = 2048

        add_block = nn.BatchNorm1d(self.feat_dim)
        add_block.apply(self._weights_init_kaiming)
        self.bn = add_block
        
        classifier = nn.Linear(self.feat_dim, num_classes)
        classifier.apply(self._weights_init_classifier)
        self.classifier = classifier

    def _weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def _weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)
            
    def _inflate_reslayer(self, reslayer2d):
        from models.STCANet3D import Bottleneck3d
        reslayers3d = []
        for layer2d in reslayer2d:
            reslayers3d.append(Bottleneck3d(layer2d))
        return nn.Sequential(*reslayers3d)
        
    def pool(self, x):
        kernel_size = x.size()[2:]
        x = F.max_pool3d(x, kernel_size=kernel_size) 
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x) 

        x2, a = self.STCABlock3D2(x)
        x = self.layer3(x2)
        x = self.layer4(x)

        x = self.pool(x)
    
        if not self.training:
            return x
      
        f = self.bn(x)
        y = self.classifier(f)  

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")

class Bottleneck3d(nn.Module):
    def __init__(self, bottleneck2d, inflate_time=False):
        super(Bottleneck3d, self).__init__()
        from models import inflate
        if inflate_time:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=3, time_padding=1, center=True)
        else:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        from models import inflate
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class STCABlock3D(nn.Module):
    def __init__(self, in_channels, seq_len=4):
        super(STCABlock3D, self).__init__()
        self.in_channels = in_channels
        conv_nd = nn.Conv3d
        bn = nn.BatchNorm3d
        self.inter_channels = in_channels // 2

        self.SA = SpatialAttn(in_channels, number=4)
        self.g = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.STAM = STIAUModule(self.inter_channels, T=seq_len)

        self.W1 = nn.Sequential(
                conv_nd(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True),
                bn(self.in_channels)
            )
        self.W2 = nn.Sequential(
                conv_nd(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True),
                bn(self.in_channels)
            )
        
        for m in self.modules():
            if isinstance(m, conv_nd) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W1[1].weight.data, 0.0)
        nn.init.constant_(self.W1[1].bias.data, 0.0)
        nn.init.constant_(self.W2[1].weight.data, 0.0)
        nn.init.constant_(self.W2[1].bias.data, 0.0)

    def apply_attention(self, x, a):
        b, c, t, h, w = x.size()
        a = a.view(a.size(0), a.size(1), h * w) 
        x = x.transpose(1, 2).contiguous().view(b * t, -1, h * w) 
        y = torch.matmul(a, x.transpose(1, 2)) 
        y = y.view(b, t, -1, c)
        return y

    def reduce_dimension(self, x, u):
        bs, t, n, c = x.size()
        x_flat = x.view(bs * t * n, c)
        u_flat = u.view(bs, 1, c).expand(bs, t*n, c).contiguous().view(bs * t * n, c)
        
        proj = nn.functional.linear(x_flat, torch.eye(self.inter_channels, c, device=x.device))
        x_out = self.g(proj.view(proj.size(0), proj.size(1), 1, 1)).view(bs, t, n, -1)
        
        u_proj = nn.functional.linear(u_flat, torch.eye(self.inter_channels, c, device=x.device))
        u_out = u_proj.view(bs * t * n, -1).mean(0, keepdim=True).expand(bs, -1)
        return x_out, u_out

    def forward(self, x):
        batch_size, t = x.size(0), x.size(2)
        g_x = x.view(batch_size, self.in_channels * t, -1) 

        theta_x = g_x 
        phi_x = g_x.permute(0, 2, 1) 
        f = torch.matmul(theta_x, phi_x) 
        f = F.softmax(f, dim=-1)

        y = torch.matmul(f, g_x) 
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        y = self.W1(y)
        z = y + x

        x = z
        inputs = x
        b, c, t, h, w = x.size()
        u = x.view(b, c, -1).mean(2) 

        a = self.SA(x) 
        a = a.transpose(1, 2).contiguous().view(b * t, -1, h, w) 

        x = self.apply_attention(x, a)
        x, u = self.reduce_dimension(x, u)
        y = self.STAM(x, u) 

        y = torch.mean(y, 2) 
        u = u.unsqueeze(1).expand_as(y)
        u = torch.cat((y, u), 2) 

        y = self.W2(u.transpose(1, 2).unsqueeze(-1).unsqueeze(-1))
        z = y + inputs
        return z, a
