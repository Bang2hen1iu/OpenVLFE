import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def apply_dropout(m):
    if m == nn.Dropout:
        m.train()

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True, top=False, use_btnk=False, ret_feat=False,  btnk_dim=256, norm=False):
        super(ResBase, self).__init__()
        self.dim = 2048
        self.top = top
        self.use_bottleneck = use_btnk
        self.return_feature = ret_feat
        self.bottleneck_dim = btnk_dim

        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)

        if top:
            self.features = model_ft
        else:
            mod = list(model_ft.children())
            mod.pop()
            self.features = nn.Sequential(*(mod[:-1]))
            self.pooling = nn.Sequential(*mod[-1:])
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(self.dim, self.bottleneck_dim)

    def forward(self, x):
        x = self.features(x)
        x_pool = self.pooling(x)
        if self.top:
            if self.return_feature:
                return x_pool, x
            return x_pool
        else:
            x_pool = x_pool.view(x_pool.size(0), self.dim)
            if self.use_bottleneck:
                x_pool = self.bottleneck(x_pool)
            if self.return_feature:
                return x_pool, x
            return x_pool
        
            

class VGGBase(nn.Module):
    def __init__(self, option='vgg', pret=True, no_pool=False, top=False):
        super(VGGBase, self).__init__()
        self.dim = 2048
        self.no_pool = no_pool
        self.top = top

        if option =='vgg11_bn':
            vgg16=models.vgg11_bn(pretrained=pret)
        elif option == 'vgg11':
            vgg16 = models.vgg11(pretrained=pret)
        elif option == 'vgg13':
            vgg16 = models.vgg13(pretrained=pret)
        elif option == 'vgg13_bn':
            vgg16 = models.vgg13_bn(pretrained=pret)
        elif option == "vgg16":
            vgg16 = models.vgg16(pretrained=pret)
        elif option == "vgg16_bn":
            vgg16 = models.vgg16_bn(pretrained=pret)
        elif option == "vgg19":
            vgg16 = models.vgg19(pretrained=pret)
        elif option == "vgg19_bn":
            vgg16 = models.vgg19_bn(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        if self.top:
            self.vgg = vgg16

    def forward(self, x):
        if self.top:
            x = self.vgg(x)
            return x
        else:
            x = self.features(x)
            x = x.view(x.size(0), 7 * 7 * 512)
            x = self.classifier(x)
            return x

class Classifier(nn.Module):
    def __init__(self, num_classes, input_size, temp=0.05, norm=True):
        super(Classifier, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if dropout:
            x = F.dropout(0.5)(x)
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)
