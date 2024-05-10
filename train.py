
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from dataloaders.ucf_dataset import UCFDataset
import torchvision.models as models
from torch.autograd import Variable


def dataloader(clip_len, batch_size, shuffle, data_type):
    root_dir = 'data/ucf101/UCF101_n_frames'
    train_list = [
        'data/ucf101/ucfTrainTestlist/trainlist01.txt'
    ]
    test_list = [
        'data/ucf101/ucfTrainTestlist/testlist01.txt'
    ]
    shuffle = shuffle

    train_dataloader = DataLoader(UCFDataset(root_dir=root_dir,
                                             info_list=train_list,
                                             split='train',
                                             clip_len=clip_len,
                                             data_type='motion_and_video'),
                                             batch_size=batch_size, shuffle=shuffle, num_workers=0)

    test_dataloader = DataLoader(UCFDataset(root_dir=root_dir,
                                            info_list=test_list,
                                            split='test',
                                            clip_len=clip_len,
                                            data_type='motion_and_video'),
                                            batch_size=batch_size, shuffle=shuffle, num_workers=0)
    #############################################
    #测试，看看输出是否正常, 如果正常请注释掉这段代码
    #############################################
    print("start test Dataloader")
    print('train_set_len: ' + str(len(train_dataloader)))
    print('test_set_len: ' + str(len(test_dataloader)))
    # for i_batch, (video, motion, labels) in enumerate(test_dataloader):
    #     print('video shape:')
    #     print(video.shape)
    #     print('motion shape:')
    #     print(motion.shape)
    #     print('labels shape')
    #     print(labels.shape)  # 每段视频的 类别
    #     break

    

    return train_dataloader, test_dataloader


########################################################
# 搭建网络, 待完成(给的代码是cifar10图像分类的卷积网络，不能直接用)
########################################################
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_headers, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_headers = num_headers
        self.num_layers = num_layers
        self.num_classes = num_classes      # 101 classes in UCF101

        # network layers
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim,
                                                      dropout=0.1,
                                                      max_len=self.input_dim)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_headers)
        self.encoder = TransformerEncoder(self.encoder_layer, self.num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1) # batch_size, frames, feature_dim
        x = self.embedding(x)
        x = self.positional_encoding(x) # bs, seq_lem, dim
        x = x.transpose(0, 1)  # 输入形状为 (seq_len, batch_size, d)
        x = self.encoder(x)  # 经过Transformer编码  
        x = x.mean(dim=0)  # 取平均作为序列表示
        x = self.fc(x)  # 全连接层分类
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

def extract_features(resnet, batch):
    features = []
    for i in range(batch.size(0)):
        video = batch[i]
        video_features = []
        for j in range(video.size(0)):
            frame = video[j]
            #frame = transform(frame)
            frame = Variable(frame.unsqueeze(0))
            if torch.cuda.is_available():
                frame = frame.cuda()
            feature = resnet(frame)
            video_features.append(feature.view(feature.size(0), -1))
        video_features = torch.cat(video_features, dim=0)
        features.append(video_features)
    features = torch.stack(features, dim=0)
    return features
########################################################
# 训练过程, 待完成
########################################################
def train(model, device, train_loader, test_loader, epochs, criterion, optimizer):

    resnet = models.resnet101(pretrained=True)
    resnet.eval()

    model.to(device)
    batches = len(train_loader)

    for epoch in range(epochs):
        model.train()   # train()模式才能调整网络参数
        batch = -1  # 记录是第几个batch
        for video, motion, labels in train_loader:
            feature_video = extract_features(resnet, video)
            feature_motion = motion.view(64, -1)
            features = torch.cat((feature_video, feature_motion), dim=1)   # 在维度 1 拼接
            batch += 1
            
            # move to device
            #motion = motion.float().to(device)
            labels = labels.to(device)

            # # print(motion.shape)
            # resnet = resnet101(pretrained=True)
            # resnet.eval()
            # motion = resnet(motion)

            # forward
            outputs = model(features)
            loss = criterion(outputs, labels)

            # record_loss
            loss_value = loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log per 10 batches
            if batch % 30 == 0: 
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch}/{batches}], "
                        f"Loss: {loss_value:.4f}")
        
        # 测试集上测试
        model.eval()       
        with torch.no_grad():
            correct = 0     # 计数分类正确的个数
            total = 0
            for motion, labels in test_loader:
                motion = motion.to(device)
                labels = labels.to(device)

                outputs = model(motion)
                # 网络输出的是一推概率(0.0~1.0)，表示网络认为这是某个类别的可能性，取可能性最大的那个类别的index
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{epochs}]", f"Test Accuracy: {accuracy:.2f}%")
            

if __name__ == '__main__':
    # 加载数据集
    batch_size = 64
    clip_len = 16
    # clip_len: 每段视频只抽取若干帧，最小的帧数好像是84，建议小于60
    # batch_size: 分批次加载 的 批大小
    # shuffle: 数据集中的数据随机排列后，再分批次加载；否则顺序加载；默认True,但需要有整个数据集；
    # data_type:
    # 如果是video,则返回video[batch_size, rgb=3, frames(=clip_len), h(图像的高)=112, w(图像的宽)=112)
    # 如果是motion,则返回video[batch_size, frames(=clip_len), 4(x,y,z,visibility))
    train_loader, test_loader = dataloader(clip_len=clip_len,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           data_type='motion_and_video')
    print('Dataloaders Ready!')     # 如果运行到这里，说明 dataloader OK

    # 指定设备：cpu or cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:')
    print(device)
    # 创建网络模型
    input_dim = 2048 + 2112  # 特征维度：33*4 #
    # [batch_size, frames, keypoints, xyzv] -> [batch_size, 时间维度， 特征维度]
    hidden_dim = 128     # transformer 隐藏层深度
    num_headers = 2     # 多注意力头，建议取值2~8
    num_layers = 2      # tf 层数，建议取值2~6
    num_classes = 101   # 分类数'''
    model = TransformerClassifier(input_dim=input_dim, hidden_dim=hidden_dim,
                                  num_headers=num_headers, num_layers=num_layers,
                                  num_classes=num_classes)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置学习率 和 训练轮数
    learning_rate = 0.0001  # 学习率，调整使得 loss逐渐收敛 且 正确率稳定上升
    epochs = 1   # 表示遍历epoch遍数据集,测试请设置为 1

    # define Loss and optimizer
    criterion = nn.CrossEntropyLoss()   # 交叉熵损失函数，常用于多分类
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # 优化器不需要改

    # 开始训练
    train(model, device, train_loader, test_loader, epochs, criterion, optimizer)
