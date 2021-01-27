import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        out = self.conv1(X) * torch.sigmoid(self.conv2(X))
        out = out.permute(0, 2, 3, 1)

        return out


class Spatial_Attention_layer(nn.Module):

    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()

        self.W1 = nn.Parameter(torch.FloatTensor(in_channels))
        self.W2 = nn.Parameter(torch.FloatTensor(num_of_timesteps, in_channels))
        self.W3 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))

    def forward(self, x, mask):
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized


class PCNLayer(nn.Module):

    def __init__(self):
        super(PCNLayer, self).__init__()
        self.fc = nn.Linear(16, 16, bias=True)

    def forward(self, x, P, roots, s2rDic, A, trajDic):

        TempDic = {}
        for r in roots:
            nodes = [r]
            while len(nodes) > 0:
                newNodes = []
                for n in nodes:
                    if n in TempDic:
                        h = TempDic[n]
                    else:
                        TempDic[n] = x[int(n), :, :, :]
                        h = x[int(n), :, :, :]
                    if P.has_key(n):
                        for c in P[n]:
                            newNodes.append(n + "/" + c)
                            propagation = self.fc(h).mul(A[:, int(n.split("/")[-1]), int(c)])
                            TempDic[n + "/" + c] = propagation + x[int(c), :, :, :]
                nodes = newNodes

        temp = {}
        for traj in TempDic:
            temp1 = []
            if not P.has_key(traj):
                lastNode = int(traj.split("/")[-1])
                if lastNode in temp:
                    temp[lastNode].append(TempDic[traj])
                else:
                    temp[lastNode] = [TempDic[traj]]

        outputs = []
        for seg in s2rDic:
            if seg in temp:
                temp2 = torch.stack(temp[seg])
                temp2 = torch.mean(temp2, dim=0)
                outputs.append(temp2)
            else:
                outputs.append(x[seg, :, :, :])
        outputs = torch.stack(outputs)
        return outputs


class PCNBlock(nn.Module):

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        super(PCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=16)
        self.PCNLayer = PCNLayer()

        self.fc = nn.Linear(32, 32, bias=True)
        self.temporal2 = TimeBlock(in_channels=64,
                                   out_channels=16)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.spatial_attention = Spatial_Attention_layer(16, 4667, 18)

    def forward(self, X, X_daily, X_weekly, P, roots, r2sDic, s2rDic, trajDic, mask):
        x = self.temporal1(X)
        x_daily = self.temporal1(X_daily)
        x_weekly = self.temporal1(X_weekly)
        x = torch.cat((x, x_daily, x_weekly), 2)
        A = self.spatial_attention(x, mask)
        x = x.permute(1, 0, 2, 3)
        tc = self.PCNLayer(x, P, roots, s2rDic, A, trajDic)
        tc = tc.permute(1, 0, 2, 3)
        return tc


class PCN(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        super(PCN, self).__init__()
        num_nodes = 4670
        num_nodes = 3505
        self.block1 = PCNBlock(in_channels=num_features, out_channels=16,
                               spatial_channels=16, num_nodes=num_nodes)
        self.block2 = PCNBlock(in_channels=64, out_channels=32,
                               spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=16)
        self.fully = nn.Linear(18 * 16,
                               num_timesteps_output, bias=True)
        self.batch_norm = nn.BatchNorm1d(num_nodes)

    def forward(self, P, roots, X, X_daily, X_weekly, r2sDic, s2rDic, trajDic, mask):
        temp = []
        tempD = []
        tempW = []
        for seg in s2rDic:
            temp.append(X[:, s2rDic[seg], :, :])
            tempD.append(X_daily[:, s2rDic[seg], :, :])
            tempW.append(X_weekly[:, s2rDic[seg], :, :])
        X = torch.stack(temp)
        X_daily = torch.stack(tempD)
        X_weekly = torch.stack(tempW)
        X = X.permute((1, 0, 2, 3))
        X_daily = X_daily.permute((1, 0, 2, 3))
        X_weekly = X_weekly.permute((1, 0, 2, 3))
        out = self.block1(X, X_daily, X_weekly, P, roots, r2sDic, s2rDic, trajDic, mask)
        out = self.fully(out.reshape((out.shape[0], out.shape[1], -1)))
        temp2 = []
        for r in r2sDic:
            temp = []
            for s in r2sDic[r]:
                temp.append(out[:, [*s2rDic].index(s), :])
            temp_new = torch.stack(temp)
            temp_new = torch.mean(temp_new, dim=0)
            temp2.append(temp_new)
        out = torch.stack(temp2).permute((1, 0, 2))
        return out


