# -*- coding: UTF-8 -*-  
# @Time : 2021/12/4 21:52
# @Author : GCR
# @File : AI_Poem2.py
# @Software : PyCharm
import numpy as np  # tang.npz的压缩格式处理
import os  # 打开文件
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchnet import meter
import tqdm


# 获取数据。
def get_data():
    if os.path.exists(data_path):
        datas = np.load(data_path, allow_pickle=True)  # 加载数据
        data = datas['data']  # numpy.ndarray
        word2ix = datas['word2ix'].item()  # dic
        ix2word = datas['ix2word'].item()  # dic
        return data, word2ix, ix2word


# 神经网络的构建
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):  # 词数量；词向量维度；隐藏层神经元数量（语境信息用256维的向量表示）
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=False)
        # lstm输入为：seq, batch
        # lstm输出为：seq * batch * 256; (2 * batch * 256,...)
        self.linear1 = nn.Linear(in_features=self.hidden_dim, out_features=vocab_size)

    def forward(self, input, hidden=None):
        # input 为 [seq_len,batch_size], 即 124x128
        seq_len, batch_size = input.size()
        # 如果没有传入 hidden，则新建
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)  # (seq_len, batch_size, embedding_dim), (124,128,128)
        # output size:[seq_len,batch_size,hidden_dim]
        output, hidden = self.lstm(embeds, (h_0, c_0))  # (seq_len, batch_size, hidden_dim), (124,128,256)
        '''
        全连接层：
        1、view的作用：将output size 从[seq_len,batch_size,hidden_dim] 变为[seq_len*batch_size,hidden_dim]
        2、output size通过这一层的变化:从[seq_len*batch_size,hidden_dim]变为[seq_len*batch_size,vocab_size]
        '''
        output = self.linear1(
            output.view(seq_len * batch_size, -1))  # ((seq_len * batch_size),hidden_dim), (15872,256) → (15872,8293)
        return output, hidden


# 训练模型
def train():
    modle = Net(len(word2ix), 128, 256)  # 模型定义：vocab_size, embedding_dim, hidden_dim —— 8293 * 128 * 256
    criterion = nn.CrossEntropyLoss()  # 计算 交叉熵损失
    if torch.cuda.is_available() == True:
        print('Cuda is available!')
        modle = modle.cuda()
        optimizer = torch.optim.Adam(modle.parameters(), lr=1e-3)  # 学习率1e-3。优化器。
        criterion = criterion.cuda()
        loss_meter = meter.AverageValueMeter()  # 计算所有数的平均数和方差。这里用来统计一个epoch中损失的平均值。

        period = []  # 记录 每个 batch 的训练的顺序。
        loss2 = []
        for epoch in range(8):  # 所有的数据训练 8 次。
            loss_meter.reset()  # 初始化
            for i, data in tqdm.tqdm(
                    enumerate(dataloader)):  # data: torch.Size([128, 125]), dtype=torch.int32。（现在是每一行表示一首诗）；训练一个 batch
                data = data.long().transpose(0, 1).contiguous()  # long为默认tensor类型，并转置, [125, 128]  （转置后，每一列表示一首诗）
                data = data.cuda()
                optimizer.zero_grad()  # 梯度置为 0

                input, target = Variable(data[:-1, :]), Variable(
                    data[1:, :])  # 数据错位。前者：第0行到倒数第二行（作为输入）  后者：第一行到最后一行（作为目标）

                output, _ = modle(input)  #
                loss = criterion(output, target.view(
                    -1))  # 计算误差  # torch.Size([15872, 8293]), torch.Size([15872])  15872=124x128
                loss.backward()  # 误差反向传播
                optimizer.step()  # 更新所有的参数

                loss_meter.add(loss.item())  # 将一个 epoch 里面产生的误差相加，求平均，作为平均误差

                period.append(i + epoch * len(dataloader))
                loss2.append(loss_meter.value()[0])

                if (1 + i) % 575 == 0:  # 每575个batch输出一次 模型的效果
                    print(str(i) + ':' + generate(modle, '床前明月光', ix2word, word2ix))

        torch.save(modle.state_dict(), 'model_poet.pth')
        plt.plot(period, loss2)
        plt.show()


# 给定诗的第一句，生成整首诗
def generate(model, start_words, ix2word, word2ix):  # 给定几个词，根据这几个词生成一首完整的诗歌
    txt = []
    for word in start_words:
        txt.append(word)
    # 手动设置第一个词为<START>
    # input初始状态:tensor([[8291]])
    input = Variable(
        torch.Tensor([word2ix['<START>']]).view(1, 1).long())  # tensor([8291.]) → tensor([[8291.]]) → tensor([[8291]])
    input = input.cuda()
    hidden = None
    num = len(txt)
    for i in range(48):  # 最大生成长度
        output, hidden = model(input, hidden)
        if i < num:
            w = txt[i]
            input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index.item()]
            txt.append(w)
            input = Variable(input.data.new([top_index])).view(1, 1)
        if w == '<EOP>':  # 遇到结束符
            break
    return ''.join(txt)


# 生成藏头诗句
def gen_acrostic(model, start_words, ix2word, word2ix):
    result = []
    txt = []
    for word in start_words:
        txt.append(word)
    input = Variable(
        torch.Tensor([word2ix['<START>']]).view(1, 1).long())  # tensor([8291.]) → tensor([[8291.]]) → tensor([[8291]])
    input = input.cuda()
    hidden = None

    num = len(txt)
    index = 0  # 用来指示已经生成了多少句藏头诗
    pre_word = '<START>'
    for i in range(48):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0]
        w = ix2word[top_index.item()]

        if (pre_word in {'。', '!', '<START>'}):
            if index == num:  # 生成的诗已经包含了全部藏头诗的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型。
                w = txt[index]
                index += 1
                input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        else:
            # 将上一次预测的词作为下一个词的输入。
            input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        result.append(w)
        pre_word = w
    return ''.join(result)


def test():
    modle = Net(len(word2ix), 128, 256)  # 模型定义：vocab_size, embedding_dim, hidden_dim —— 8293 * 128 * 256
    if torch.cuda.is_available() == True:
        modle.cuda()
        modle.load_state_dict(torch.load('model_poet.pth'))
        modle.eval()
        txt = generate(modle, '床前明月光', ix2word, word2ix)
        print(txt)
        txt = gen_acrostic(modle, '深度学习', ix2word, word2ix)
        print(txt)


if __name__ == '__main__':
    data_path = 'tang.npz'
    data, word2ix, ix2word = get_data()
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True, num_workers=1)  # shuffle=True随机打乱
    # train()
    test()
