import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_improve import BD_BGL  # 导入修改后的模型
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--id_feature_size', type=int, default=3)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--lr', type=list, default=[0.003, 0.0003, 0.0003, 0.0003])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--highlow', type=int, default=0)
parser.add_argument('--epoches', type=int, default=200)
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--feat_drop', type=float, default=0.3)
parser.add_argument('--attn_drop', type=float, default=0.7)
args = parser.parse_args()

def main():

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = args.lr

    # for fraud_type in range(-1,3):

    train_id = np.load(
        '/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/train_ids.npy')
    test_id = np.load(
        '/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/test_ids.npy')

    num_node = len(train_id)+len(test_id)
    train_index = len(train_id)

    oneweek = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_train_1.npy')
    twoweek = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_train_2.npy')
    threeweek = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_train_3.npy')
    month = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_train_4.npy')
    id_feature = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/id_x_train.npy')
    label = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/y_train.npy')

    oneweek = np.nan_to_num(oneweek)
    twoweek = np.nan_to_num(twoweek)
    threeweek = np.nan_to_num(threeweek)
    month = np.nan_to_num(month)
    id_feature = np.nan_to_num(id_feature)

    oneweek = torch.from_numpy(oneweek).type(torch.float32).cuda()
    twoweek = torch.from_numpy(twoweek).type(torch.float32).cuda()
    threeweek = torch.from_numpy(threeweek).type(torch.float32).cuda()
    month = torch.from_numpy(month).type(torch.float32).cuda()
    id_feature = torch.from_numpy(id_feature).type(torch.float32).cuda()

    oneweek2 = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_test_1.npy')
    twoweek2 = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_test_2.npy')
    threeweek2 = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_test_3.npy')
    month2 = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/x_test_4.npy')
    id_feature2 = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/id_x_test.npy')
    label2 = np.load(
        f'/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/y_test.npy')

    oneweek2 = np.nan_to_num(oneweek2)
    twoweek2 = np.nan_to_num(twoweek2)
    threeweek2 = np.nan_to_num(threeweek2)
    month2 = np.nan_to_num(month2)
    id_feature2 = np.nan_to_num(id_feature2)

    oneweek2 = torch.from_numpy(oneweek2).type(torch.float32).cuda()
    twoweek2 = torch.from_numpy(twoweek2).type(torch.float32).cuda()
    threeweek2 = torch.from_numpy(threeweek2).type(torch.float32).cuda()
    month2 = torch.from_numpy(month2).type(torch.float32).cuda()
    id_feature2 = torch.from_numpy(id_feature2).type(torch.float32).cuda()

    oneweek = torch.concat([oneweek, oneweek2], dim=0)
    twoweek = torch.concat([twoweek, twoweek2], dim=0)
    threeweek = torch.concat([threeweek, threeweek2], dim=0)
    month = torch.concat([month, month2], dim=0)
    id_feature = torch.concat([id_feature, id_feature2], dim=0)

    source = np.load(
        '/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/source_1p.npy')
    target = np.load(
        '/content/drive/MyDrive/bs/fraud_detection_gnn/data_processing/output2/target_1p.npy')
    tmps = source.copy()
    tmpt = target.copy()

    source = np.concatenate([source, tmpt])
    target = np.concatenate([target, tmps])
    edge_index = np.stack([source, target], axis=0)
    edge_index = torch.from_numpy(edge_index).cuda()
    # data=torch_geometric.data.Data()
    device = torch.device('cuda:'+args.device)

    label = torch.from_numpy(label).type(torch.long).cuda()
    label2 = torch.from_numpy(label2).type(torch.long).cuda()
    # # torch.Size([6106])
    label = torch.concat([label, label2], dim=0)

    model = BD_BGL(args, edge_index, label).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    gamma = args.gamma

    gamma = round(gamma, 1)
    losses = list()
    accs = list()
    xlabel = []
    for epoch in range(args.epoches):
        dual_pre, multilstm_pre = model(
            oneweek, twoweek, threeweek, month, id_feature,)
        optimizer.zero_grad()
        # loss_v=loss(pre[:train_index],label[:train_index])
        loss_v2 = loss(multilstm_pre[:train_index], label[:train_index])
        # loss_v3=loss_v+loss_v2
        loss_v3 = F.nll_loss(dual_pre[:train_index], label[:train_index])
        a = torch.tensor([gamma]).cuda()
        b = torch.tensor([1-gamma]).cuda()
        loss_v = gamma*loss_v2+(1-gamma)*loss_v3
        loss_v.backward()
        losses.append(loss_v.item())
        xlabel.append(epoch)
        optimizer.step()

        model.eval()
        dual_pre, multilstm_pre = model(
            oneweek, twoweek, threeweek, month, id_feature, )

        test_pre = dual_pre[train_index:]
        test_label = label[train_index:]

        test_pre = test_pre.detach().cpu().numpy()
        test_label = test_label.detach().cpu().numpy()
        test_pre_y = np.argmax(test_pre, axis=-1)
        pos_score = test_pre[:, 1]
        acc = accuracy_score(test_label, test_pre_y)
        prec = precision_score(test_label, test_pre_y, average='macro')
        recall = recall_score(test_label, test_pre_y, average='macro')
        f1 = f1_score(test_label, test_pre_y, average='macro')
        auc = roc_auc_score(test_label, pos_score)

        accs.append(acc)

    print(
        f'acc:{acc} prec:{prec} recall:{recall} f1:{f1} auc:{auc}')



    torch.save(model, '/content/drive/MyDrive/bs/fraud_detection_gnn/model_output/BD-BGL_improve.pt')
    plt.cla()
    plt.plot(xlabel, losses,)
    plt.savefig('/content/drive/MyDrive/bs/fraud_detection_gnn/loss_picture/BD-BGL_improve.png')
    plt.cla()
    plt.plot(xlabel, accs, )
    plt.savefig('/content/drive/MyDrive/bs/fraud_detection_gnn/acc_picture/BD-BGL_improve.png')

    # model.eval()
    # dual_pre, multilstm_pre = model(
    #     oneweek, twoweek, threeweek, month, id_feature,)
    #
    # test_pre = dual_pre[train_index:]
    # test_label = label[train_index:]
    #
    # test_pre = test_pre.detach().cpu().numpy()
    # test_label = test_label.detach().cpu().numpy()
    # test_pre_y = np.argmax(test_pre, axis=-1)
    # pos_score = test_pre[:, 1]
    # acc = accuracy_score(test_label, test_pre_y)
    # prec = precision_score(test_label, test_pre_y, average='macro')
    # recall = recall_score(test_label, test_pre_y, average='macro')
    # f1 = f1_score(test_label, test_pre_y, average='macro')
    # auc = roc_auc_score(test_label, pos_score)
    # print(
    #     f'acc:{acc} prec:{prec} recall:{recall} f1:{f1} auc:{auc}')


if __name__ == '__main__':
    main()


