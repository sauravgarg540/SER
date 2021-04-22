# #!/usr/bin/env python
# # coding: utf-8


# import torch
# import os
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim

##Model by Ma
# class AlexNet(nn.Module):
#     def __init__(self,num_class=5):
#         super().__init__()
#         #（1,f=200,t=960*x）
#         self.block1=nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2,groups=2),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.block3 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3,stride=1),
#             nn.ReLU(inplace=True)
#         )

#         self.block4 = nn.Sequential(
#             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1,stride=1),
#             nn.ReLU(inplace=True)
#         )

#         self.block5 = nn.Sequential(
#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1,stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         #(256,F,T)
#         self.MLP = nn.Sequential(
#             nn.Linear(in_features=256, out_features=256),
#             nn.Tanh()
#         )

#         self.ei=nn.Linear(in_features=256,out_features=256,bias=False)
#         self.classification=nn.Linear(in_features=256,out_features=num_class)


#     def forward(self, x):
#         lbda=0.3
#         x=self.block1(x)
#         x=self.block2(x)
#         x=self.block3(x)
#         x=self.block4(x)
#         x=self.block5(x)

#         x=x.permute([0,3,2,1])
#         tmp=x.size()[1]*x.size()[2]
#         x=x.contiguous().view([-1,tmp,256])
#         e=self.ei(self.MLP(x))
#         e=F.softmax(torch.mul(e, lbda),dim=1)
#         x=x.mul(e)

#         x=torch.sum(x, dim=-2)
#         x=F.softmax(self.classification(x))
#         return x
    # #!/usr/bin/env python
# # coding: utf-8
#
##Model by Saurav
import torch.nn as nn
import torchvision.models as models
import os
import torch
import torch.nn.functional as F


num_class = 5


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh()
        )
        self.ei = nn.Linear(in_features=256, out_features=256, bias=False)

    def forward(self, x):
        #print(x.size())
        lbda = 0.3
        x = x.permute([0, 3, 2, 1])
        tmp = x.size()[1] * x.size()[2]
        x = x.contiguous().view([-1, tmp, 256])
        e = self.ei(self.mlp(x))
        e = F.softmax(torch.mul(e, lbda), dim=1)
        x = x.mul(e)
        # print("before" , x.size())
        return x


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.classification = nn.Linear(in_features=256, out_features=num_class)

    def forward(self, x):
        x = torch.sum(x, dim=-2)
        x = F.softmax(self.classification(x))
        return x


def AlexNet():

    os.environ['TORCH_HOME'] = '/home/zzhang/test/'
    alexnet = models.alexnet(pretrained=True)

    alexnet.features[0] = nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4))
    alexnet.features[1] = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0))
    alexnet.features[3] = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2,2), groups= 2)
    alexnet.features[4] = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0))
    alexnet.features[6] = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
    alexnet.features[8] = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
    alexnet.features[10] = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
    alexnet.avgpool = MLP()
    alexnet.classifier = Classification()

    return alexnet
