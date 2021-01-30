from model import PoetryModel
from config import Config
import torch
from torch.utils.data import DataLoader
from sample import generate, gen_acrostic
from dataHandler import load_data
import os

def train():
    if Config.use_gpu and torch.cuda.is_available():
        Config.device = torch.device("cuda:0")
    else:
        Config.device = torch.device("cpu")
    device = Config.device
    # 获取数据
    data, char2ix, ix2char = load_data()
    data = torch.from_numpy(data)
    data_loader = DataLoader(data,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=1)

    # 定义模型
    model = PoetryModel(len(char2ix),
                        embedding_dim=Config.embedding_dim,
                        hidden_dim=Config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 转移到相应计算设备上
    model.to(device)
    # 进行训练
    for epoch in range(Config.epoch):
        for step, data_ in enumerate(data_loader):
            # print(data_.shape)
            data_ = data_.long().transpose(1, 0).contiguous()
            # 注意这里，也转移到了计算设备上
            data_ = data_.to(device)
            optimizer.zero_grad()
            # n个句子，前n-1句作为输入，后n-1句作为输出，二者一一对应
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            # 进行可视化
            if (1 + step) % Config.plot_every == 0:
                save_path = os.path.join(Config.model_root, "tang_%s_%s.pth" % (epoch,step))
                print("%s loss: %f" % (save_path,loss.data))
                for word in list(u"天青色等烟雨"):
                    gen_poetry = ''.join(gen_acrostic(model, word, ix2char, char2ix))
                    print(gen_poetry)
                torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    train()
