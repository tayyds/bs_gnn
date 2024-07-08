import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax


# LSTM的时间序列模型
'''
self.embedding: 一个可训练的参数矩阵，用于将输入数据嵌入到指定的隐藏层维度中。
self.feature_attn: 一个可训练的参数向量，用于对输入特征进行注意力加权。
self.weekLstm 到 self.monthLstm: 四个LSTM层，用于处理不同时间范围的数据（如一周、两周、三周和一个月）。
self.weekmonthattn: 一个可训练的参数向量，用于对不同时间范围的LSTM输出进行注意力加权。
self.softmax: 用于对特征和时间范围进行软最大归一化。
self.clf: 一个线性层，用于将LSTM的输出映射到分类输出
'''


class MultiLstm(nn.Module):
    def __init__(self, arg) -> None:
        super().__init__()
        # self.device=torch.device(arg.device)

        # 定义一个可训练的参数 self.embedding，其大小为 (1, arg.hidden_size)，并用随机值初始化   [1 行和 arg.hidden_size 列]
        self.embedding = Parameter(torch.rand(size=(1, arg.hidden_size), ), requires_grad=True)

        # 定义另一个可训练的参数 self.feature_attn，其大小为 arg.num_features，同样用随机值初始化
        # arg.num_features 表示输入数据的特征数量
        self.feature_attn = Parameter(torch.rand(
            size=(arg.num_features,), ), requires_grad=True)
        # 定义四个LSTM层，分别用于处理不同时间范围的数据（每周、两周、三周、每月）。
        # 每个LSTM的输入和输出维度均为 arg.hidden_size，并设置 batch_first=True 使批次大小为第一维度
        '''
        input_size：这是输入特征的维度。在LSTM中，每个时间步的输入数据通常是一个向量，input_size 就是这个向量的维度
        hidden_size：这是LSTM隐藏状态的维度。LSTM的每个单元有一个隐藏状态和一个单元状态，
        hidden_size 就是隐藏状态向量的维度，也是LSTM输出的维度 记忆细胞个数
        '''
        self.weekLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.twoweekLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.threeweekLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.monthLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        # 定义一个可训练的参数 self.weekmonthattn，其大小为 4，用于对不同时间范围的LSTM输出进行加权
        self.weekmonthattn = Parameter(
            torch.rand(size=(4,)), requires_grad=True)
        # 定义一个Softmax层
        # dim=-1 表示在最后一个维度上应用 softmax 操作
        self.softmax = nn.Softmax(dim=-1)
        # 定义一个线性层 self.clf，将拼接后的特征映射到分类输出。输入大小为 arg.hidden_size*4 + arg.id_feature_size，输出大小为 2
        self.clf = nn.Linear(arg.hidden_size * 4 + arg.id_feature_size, 2)
        # self.fagcn=FAGCN()

    '''
        input_week、input_twoweek、input_threeweek 和 input_month 原始形状均为 (batch_size, sequence_length, feature_dim)
        简写成即 (B, S, F)
        使用torch.unsqueeze方法是增加维度 dim=-1表示在最后一个维度新增一个维度 方便于embdding进行矩阵乘法运算
    '''

    def forward(self, input_week, input_twoweek, input_threeweek, input_month, id_feature=None):
        '''
        插入新维度的目的：插入新维度后，可以与形状为 (1, embedding_dim) 的嵌入矩阵进行矩阵乘法，而不需要手动调整张量的形状
        新的维度的索引为-1，表示最后一个维度。通过增加这个维度，可以进行后续的矩阵乘法运算
        '''
        # 在 input_week 的最后一个维度（即 dim=-1）插入一个新维度 (B, S, F, 1)
        input_week = torch.unsqueeze(input_week, dim=-1)
        input_twoweek = torch.unsqueeze(input_twoweek, dim=-1)
        input_threeweek = torch.unsqueeze(input_threeweek, dim=-1)
        input_month = torch.unsqueeze(input_month, dim=-1)

        # torch.matmul 矩阵乘法
        '''
        input_week形状(B, S, F，1)  embedding形状 (F, E) F 是输入特征的维度。  embedding形状 (1, E)
                                                        E 是嵌入后的特征维度。 其中F在上面init方法可知F=1
            这两个相乘得到(B, S, F，E)
        '''
        input_week = torch.matmul(input_week, self.embedding)
        # @ 符号用于执行矩阵乘法运算，相当于 torch.matmul 函数
        input_twoweek = input_twoweek @ self.embedding
        input_threeweek = input_threeweek @ self.embedding
        input_month = input_month @ self.embedding
        '''
            torch.permute函数 对维度进行变换
            维度 (0, 1, 3, 2) 将2和3维度进行交换  input_week 形状(B, S, F，E)  permute后 --》 (B, S, E，F)
            目的是将置换后的 input_week 张量与经过 softmax 函数处理后的 self.feature_attn 矩阵进行矩阵乘法

            因为上面定义的feature_attn的size=(arg.num_features,)  arg.num_features是输入特征数量，即F

            self.softmax(self.feature_attn) 的形状为 (F,)。
            矩阵乘法 @ 操作会自动将 self.softmax(self.feature_attn) 视为 (F, 1)，因此乘法操作实际上是 (B, S, 1, F) @ (F, 1)
        '''
        input_week = torch.permute(input_week, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)
        input_twoweek = torch.permute(input_twoweek, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)
        input_threeweek = torch.permute(input_threeweek, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)
        input_month = torch.permute(input_month, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)

        # 调用LSTM网络模型 这里没有给h0和c0 默认为0
        output, (final_week, _) = self.weekLstm(input_week)
        output2, (final_twoweek, _) = self.weekLstm(input_twoweek)
        output3, (final_threeweek, _) = self.weekLstm(input_threeweek)
        output4, (final_month, _) = self.weekLstm(input_month)

        # final_week 的形状是 (num_layers, batch_size, hidden_size)
        '''
            假设 final_week 的原始形状是 (N, B, H)，经过索引操作 final_week[0, :, :] 后，形状将变为 (B, H)，
            即 (batch_size, hidden_size)。 N：num_layers

            final_week[0, :, :] 提取了最后一个时间步的隐藏状态，并移除了表示层数的维度，形状为 (B, H)  
        '''
        final_week = final_week[0, :, :]
        final_twoweek = final_twoweek[0, :, :]
        final_threeweek = final_threeweek[0, :, :]
        final_month = final_month[0, :, :]
        # final=torch.stack([final_week,final_twoweek,final_threeweek,final_month],dim=-1)@self.weekmonthattn

        # 拼接
        # 拼接后的 final 的形状为 (batch_size, hidden_size * 4 + id_feature_size)。
        final = torch.concat(
            [final_week, final_twoweek, final_threeweek, final_month, id_feature], dim=-1)
        # 将final得到的特征传入clf 最后得到2分类结果
        final2 = self.clf(final)
        final2 = self.softmax(final2)
        return final2, final

        # return final

    # 对self.feature_attn进行softmax
    def get_fea_attn(self):
        return self.softmax(self.feature_attn)

    # 对self.weekmonthattn进行softmax
    def get_timeattn(self):
        return self.softmax(self.weekmonthattn)



# class DiffAttention(MessagePassing):
#     # in_dim 258 out_dim 64
#     def __init__(self, in_dim, out_dim, edge_index, feat_drop=0, attn_drop=0, device='cuda'):
#         super(DiffAttention, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.edge_index = edge_index
#         self.device = device
#
#
#         self.fc = nn.Linear(in_dim, out_dim, bias=False)
#         nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
#
#         self.attn_fc = nn.Linear(out_dim, 1, bias=False)
#         nn.init.xavier_uniform_(self.attn_fc.weight, gain=1.414)
#
#     def edge_attention(self, x_i, x_j):
#         diff = x_j - x_i
#         diff = self.fc(self.feat_drop(diff.float()))
#         a = self.attn_fc(diff)
#         return torch.tanh(a), diff
#
#     def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
#         a, diff = self.edge_attention(x_i, x_j)
#         alpha = softmax(a, index, ptr, size_i)
#         alpha = self.attn_drop(alpha)
#         return diff * alpha
#
#     # x形状(6106, 64)
#     def forward(self, x, edge_index):
#         # Separate source and target node features
#         # 得到起始点和终点的特征 即 邻居节点和中心节点的特征
#         # h_src形状(num_edges, features)=(num_edges, 64)=(382622,64)
#         h_src, h_dst = x[edge_index[0]], x[edge_index[1]]
#
#         # Apply message passing
#         out = self.propagate(edge_index, x=(h_src, h_dst))
#         return F.elu(out)

# class DualChannelLayer(nn.Module):
#     # num_hidden 节点特征
#     def __init__(self, num_hidden, edge_index, label, dropout, highlow,in_dim, out_dim):
#         # 聚合操作用add加法
#         super(DualChannelLayer, self).__init__()
#         # 边信息
#         self.edge_index = edge_index
#         self.label = label
#         self.dropout = nn.Dropout(dropout)
#         self.highlow = highlow
#         self.diff_attention = DiffAttention(in_dim, out_dim, feat_drop=dropout, attn_drop=dropout)
#
#
#         # 线性层 输入维度2 * num_hidden 输出维度1
#         '''
#         2 * num_hidden:图神经网络中，每条边连接两个节点。为了计算这条边的重要性或权重，我们通常会考虑这两个节点的特征
#         '''
#         self.gate = nn.Linear(2 * num_hidden, 1)
#
#         # 从边索引中提取的行和列，用于表示每条边的起点和终点节点
#         '''
#         edge_index 的形状是 (2, num_edges)
#         self.row：表示边的起点节点的索引
#         self.col: 表示边的终点节点的索引
#         '''
#         self.row, self.col = edge_index
#
#         self.norm_degree = degree(self.row, num_nodes=label.shape[0]).clamp(min=1)
#
#         self.norm_degree = torch.pow(self.norm_degree, -0.5)
#
#         self.highlow = highlow
#
#         nn.init.xavier_normal_(self.gate.weight, gain=1.414)
#
#     # 节点特征矩阵 h
#     def forward(self, h):
#         h2 = torch.cat([h[self.row], h[self.col]], dim=1)
#         # h2 = h[self.row] - h[self.col]
#         g = torch.tanh(self.gate(h2)).squeeze()
#
#         if self.highlow == 1:
#             temp = torch.ones_like(g, dtype=g.dtype).to(g.device)
#             g = temp
#         elif self.highlow == -1:
#             temp = torch.ones_like(g, dtype=g.dtype).to(g.device)
#             temp = -temp
#             g = temp
#
#         norm = g * self.norm_degree[self.row] * self.norm_degree[self.col]
#         norm = self.dropout(norm)
#         '''
#         self.propagate 是 MessagePassing 基类中的一个方法。这个方法负责在图神经网络中执行消息传递和特征聚合
#         self.edge_index：这是边索引矩阵，形状为 (2, num_edges)，
#         其中 self.edge_index[0] 是边的起点节点索引，self.edge_index[1] 是边的终点节点索引
#
#         size=(h.size(0), h.size(0))：size 参数指定了输入和输出的节点数量。
#         在这里，h.size(0) 是节点的数量，表示输入和输出的节点数量相同
#
#         x 是节点特征矩阵，形状为 (num_nodes, num_hidden)，其中 num_nodes 是节点数量，num_hidden 是每个节点特征的维度
#         '''
#         return self.propagate(self.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)
#
#     # message函数的作用就是对邻居信息的变换映射
#     # x_j是邻居节点 x_i是中心节点
#     # norm.view(-1, 1) * x_j 将归一化边权重与终点节点特征相乘，形成消息
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j
#
#     # 用于更新节点特征。将聚合后的消息用于更新节点特征
#     def update(self, aggr_out):
#         return aggr_out

# class DiffAttention(MessagePassing):
#     def __init__(self, in_dim, out_dim, feat_drop=0, attn_drop=0, device='cuda'):
#         super(DiffAttention, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.fc0 = nn.Linear(out_dim, in_dim, bias=False)
#         self.fc = nn.Linear(in_dim, out_dim, bias=False)
#         self.attn_fc = nn.Linear(out_dim, 1, bias=False)
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
#         nn.init.xavier_uniform_(self.attn_fc.weight, gain=1.414)
#         self.device = device
#     # (6106,64)
#     def forward(self, x, edge_index):
#         x = self.feat_drop(x)
#         out = self.propagate(edge_index, x=x)
#         return F.elu(out)
#
#     def message(self, x_i, x_j, edge_index, size):
#         diff = x_j - x_i
#         diff = self.fc0(diff)
#         diff = self.fc(diff)
#         e = self.attn_fc(diff)
#         return {'e': torch.tanh(e), 'diff': diff}
#
#     def aggregate(self, inputs, index, dim_size=None):
#         e = inputs['e']
#         diff = inputs['diff']
#         alpha = softmax(e, index, num_nodes=dim_size)
#         alpha = self.attn_drop(alpha)
#         return torch.sum(alpha * diff, dim=1)


class MultiGRU(nn.Module):
    def __init__(self, arg) -> None:
        super().__init__()
        # self.device=torch.device(arg.device)

        # 定义一个可训练的参数 self.embedding，其大小为 (1, arg.hidden_size)，并用随机值初始化   [1 行和 arg.hidden_size 列]
        self.embedding = Parameter(torch.rand(size=(1, arg.hidden_size), ), requires_grad=True)

        # 定义另一个可训练的参数 self.feature_attn，其大小为 arg.num_features，同样用随机值初始化
        # arg.num_features 表示输入数据的特征数量
        self.feature_attn = Parameter(torch.rand(
            size=(arg.num_features,), ), requires_grad=True)
        # 定义四个LSTM层，分别用于处理不同时间范围的数据（每周、两周、三周、每月）。
        # 每个LSTM的输入和输出维度均为 arg.hidden_size，并设置 batch_first=True 使批次大小为第一维度
        self.weekGRU = nn.GRU(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.twoweekGRU = nn.GRU(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.threeweekGRU = nn.GRU(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.monthGRU = nn.GRU(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        # 定义一个可训练的参数 self.weekmonthattn，其大小为 4，用于对不同时间范围的LSTM输出进行加权
        self.weekmonthattn = Parameter(
            torch.rand(size=(4,)), requires_grad=True)

        self.clf = MLP(arg.hidden_size * 4 + arg.id_feature_size, arg.hidden_size, 2)

        # 定义一个Softmax层
        # dim=-1 表示在最后一个维度上应用 softmax 操作
        self.softmax = nn.Softmax(dim=-1)
        # 定义一个线性层 self.clf，将拼接后的特征映射到分类输出。输入大小为 arg.hidden_size*4 + arg.id_feature_size，输出大小为 2
        # self.clf = nn.Linear(arg.hidden_size * 4 + arg.id_feature_size, 2)



    def forward(self, input_week, input_twoweek, input_threeweek, input_month, id_feature=None):
        '''
        插入新维度的目的：插入新维度后，可以与形状为 (1, embedding_dim) 的嵌入矩阵进行矩阵乘法，而不需要手动调整张量的形状
        新的维度的索引为-1，表示最后一个维度。通过增加这个维度，可以进行后续的矩阵乘法运算
        '''
        # 在 input_week 的最后一个维度（即 dim=-1）插入一个新维度 (B, S, F, 1)
        input_week = torch.unsqueeze(input_week, dim=-1)
        input_twoweek = torch.unsqueeze(input_twoweek, dim=-1)
        input_threeweek = torch.unsqueeze(input_threeweek, dim=-1)
        input_month = torch.unsqueeze(input_month, dim=-1)

        # torch.matmul 矩阵乘法
        input_week = torch.matmul(input_week, self.embedding)
        # @ 符号用于执行矩阵乘法运算，相当于 torch.matmul 函数
        input_twoweek = input_twoweek @ self.embedding
        input_threeweek = input_threeweek @ self.embedding
        input_month = input_month @ self.embedding

        input_week = torch.permute(input_week, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)
        input_twoweek = torch.permute(input_twoweek, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)
        input_threeweek = torch.permute(input_threeweek, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)
        input_month = torch.permute(input_month, dims=(
            0, 1, 3, 2)) @ self.softmax(self.feature_attn)

        # 调用LSTM网络模型 这里没有给h0和c0 默认为0
        output, final_week = self.weekGRU(input_week)
        output2, final_twoweek = self.twoweekGRU(input_twoweek)
        output3, final_threeweek = self.threeweekGRU(input_threeweek)
        output4, final_month = self.monthGRU(input_month)

        # final_week 的形状是 (num_layers, batch_size, hidden_size)
        final_week = final_week[0, :, :]
        final_twoweek = final_twoweek[0, :, :]
        final_threeweek = final_threeweek[0, :, :]
        final_month = final_month[0, :, :]
        # final=torch.stack([final_week,final_twoweek,final_threeweek,final_month],dim=-1)@self.weekmonthattn

        # 拼接
        # 拼接后的 final 的形状为 (batch_size, hidden_size * 4 + id_feature_size)。
        final = torch.concat(
            [final_week, final_twoweek, final_threeweek, final_month, id_feature], dim=-1)
        # 将final得到的特征传入clf 最后得到2分类结果
        final2 = self.clf(final)
        final2 = self.softmax(final2)
        return final2, final

        # return final

    # 对self.feature_attn进行softmax
    def get_fea_attn(self):
        return self.softmax(self.feature_attn)

    # 对self.weekmonthattn进行softmax
    def get_timeattn(self):
        return self.softmax(self.weekmonthattn)


class MLP(nn.Module):
    def __init__(self, in_size, hidden, out_size, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_size, 2*hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, out_size)
        )
    def forward(self,feat):
        return self.mlp(self.dropout(feat))


class DualChannelLayer(MessagePassing):
    # num_hidden=64  edge_index (2, 382622)  label torch.Size([6106])
    def __init__(self, num_hidden, edge_index, label, dropout, highlow):
        # 聚合操作用add加法
        super(DualChannelLayer, self).__init__(aggr='mean')
        # self.data = data
        # 边信息
        self.edge_index = edge_index
        self.label = label
        self.dropout = nn.Dropout(dropout)
        # 线性层 输入维度2 * num_hidden 输出维度1
        '''
        2 * num_hidden:图神经网络中，每条边连接两个节点。为了计算这条边的重要性或权重，我们通常会考虑这两个节点的特征
        '''
        # num_hidden这里等于64 2 * num_hidden=128
        self.gate = nn.Linear(2 * num_hidden, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

        self.highlow = highlow



    # 节点特征矩阵 h 形状(6106, 64)
    def forward(self, h, edge_index):
        # 从边索引中提取的行和列，用于表示每条边的起点和终点节点
        # row=col=382622  形状是(382622,)
        self.row, self.col = edge_index
        self.node_degree = degree(self.row, num_nodes=h.size(0)).clamp(min=1)
        self.norm_degree = torch.pow(self.node_degree, -0.5)

        # h2形状(382622, 128) 拼接节点
        h2 = torch.cat([h[self.row], h[self.col]], dim=1)

        # g的形状(382622,1) 使用squeeze后 g形状(382622,)
        g = torch.tanh(self.gate(h2)).squeeze()

        '''
        这段代码的作用是根据 self.highlow 参数的值，对边的权重 g 进行特殊处理，将其全部设置为 1 或 -1
        torch.ones_like(g, dtype=g.dtype) 创建一个和 g 形状一样且元素全为 1 的张量，并且数据类型与 g 相同

        '''
        norm = g * self.norm_degree[self.row] * self.norm_degree[self.col]
        norm = self.dropout(norm)

        return self.propagate(self.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    # message函数的作用就是对邻居信息的变换映射
    # x_j是邻居节点 x_i是中心节点
    # norm.view(-1, 1) * x_j 将归一化边权重与终点节点特征相乘，形成消息

    def message(self, x_i, x_j, norm):
        combine_feature = x_i + x_j
        return norm.view(-1, 1) * combine_feature

    # 用于更新节点特征。将聚合后的消息用于更新节点特征
    def update(self, aggr_out):
        return aggr_out


class DualChannel(nn.Module):
    # num_features 258  num_hidden  64   num_classes 2
    def __init__(self, num_features, num_hidden, num_classes, edge_index, labels, dropout, eps, layer_num=2, highlow=0):
        super(DualChannel, self).__init__()
        # self.eps = eps：保存用于残差连接的缩放因子
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.edge_index = edge_index
        self.layers = nn.ModuleList()

        # 迭代 layer_num 次
        # 每次迭代添加一个 DualChannelLayer 层到 self.layers 中，使用相同的参数 num_hidden, edge_index, labels, dropout, 和 highlow
        for i in range(self.layer_num):
            self.layers.append(DualChannelLayer(num_hidden, edge_index, labels, dropout, highlow))
        # num_features 258   num_hidden 64
        self.t1 = nn.Linear(num_features, num_hidden)
        # num_hidden 64  num_classes 2
        self.t2 = nn.Linear(num_hidden, num_classes)
        self.reset_parameters()

    # 初始化权重方法
    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    # h 形状 (6106, 258)
    def forward(self, h):
        # self.training 是 nn.Module 的属性，当调用 model.train() 时，它被设置为 True；当调用 model.eval() 时，它被设置为 False
        h = F.dropout(h, p=self.dropout, training=self.training)
        # h (6106, 258) 经过t1这个线性层后变成(6106, 64)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # 将当前特征保存到 raw，用于后续的残差连接
        raw = h
        # 通过迭代方式将特征 h 传递给每个 DualChannelLayer 层，并应用残差连接（Residual Connection）
        for i in range(self.layer_num):
            # 将特征 h 传递给第 i 个 DualChannelLayer 层，得到更新后的特征 h。 这里的h形状(6106, 64)
            h = self.layers[i](h,self.edge_index)
            # 残差链接
            h = self.eps * raw + h
        # 这里输入时的h形状发生改变了 加了一个残差连接 但输出经过一个t2线性层 h形状变成(6106,2)
        h = self.t2(h)
        # dim：指定进行 log-softmax 操作的维度。在这个例子中，值为 1，表示在第二个维度（通常是特征维度）上进行操作
        return F.log_softmax(h, 1)