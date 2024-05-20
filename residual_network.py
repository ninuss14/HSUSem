import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=43):
        self.inplanes = 7
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(7)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(block, 7, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 9, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 15, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 20, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 23, layers[4], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.45)
        self.fc = nn.Linear(23 * block.expansion, num_classes)

        # inicializacia vah, bias pre vsetky vrstvy
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # menime velkost len ak nam nesedia input a output
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
