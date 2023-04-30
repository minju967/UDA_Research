import torch.nn.functional as F
import torch.nn as nn
import torch


class MI_caculate(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
        self.c1 = nn.Conv2d(32, 32, kernel_size=4, stride=2)

        self.l0 = nn.Linear(32 * 26 * 26 + 256, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, f_map, vector):
        M = f_map
        y = vector

        h = F.relu(self.c0(M))
        h = F.relu(self.c1(h))       #[batch, 32, 26, 26]
        h = h.view(y.shape[0], -1)   #[batch, 32*26*26]
        h = torch.cat((y, h), dim=1) #[batch, 32*26*26+256]
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)
        
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 65)
        self.bn2 = nn.BatchNorm1d(65)
        self.l3 = nn.Linear(65, 65)
        self.bn3 = nn.BatchNorm1d(65)

    def forward(self, vector):
        content = vector
        # clazz = F.relu(self.bn1(self.l1(content)))
        # clazz = F.relu(self.bn2(self.l2(clazz)))
        # clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        clazz = F.relu(self.l1(content))
        clazz = F.relu(self.l2(clazz))
        clazz = F.softmax(self.l3(clazz), dim=1)
        return clazz

class Classifier_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.l3 = nn.Linear(512, 65)
        self.bn3 = nn.BatchNorm1d(65)

    def forward(self, vector):
        content = vector
        clazz = F.relu(self.l1(content))
        clazz = F.relu(self.l2(clazz))
        clazz = F.softmax(self.l3(clazz), dim=1)
        return clazz


class Loss(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args       = args
        self.MI_val     = MI_caculate()
        self.classifier = Classifier()
        
    def forward(self, f_map, c_vec, d_vec, label):
        Content = 0
        Domain  = 0
        Class   = 0 
        Acc     = dict()

        for dset in self.args.datasets:
            correct = 0
            total   = 0
            c = F.sigmoid(self.MI_val(f_map[dset], d_vec[dset]))
            Content += torch.log(-(1/(c -1))).mean()
            d = F.sigmoid(self.MI_val(f_map[dset], c_vec[dset]))        
            Domain  += torch.log(1/d).mean()

            output = self.classifier(c_vec[dset])
            Class  += F.cross_entropy(output, label[dset])

            _, pred = torch.max(output.data, 1)
            total   += label[dset].size(0)
            correct += (pred == label[dset]).sum().item()
            
            Acc[dset] = 100 * (correct//total)

        print(f'Content_MI: {Content:.4f}, Domain_MI: {Domain:.4f}, Cls_loss: {Class:.4f}')
        loss = Content + Domain + Class
        return loss, Acc