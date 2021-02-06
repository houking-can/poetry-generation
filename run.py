import argparse
import re

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model


def train(dataset, model, args):
    # 设置为训练模式
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )
    for epoch in range(args.max_epochs):
        if (args.previous != 0) and (epoch <= args.previous):
            print('continue...', epoch)
            continue

        for batch, (input, label) in enumerate(data_loader):
            input, label = input.to(device), label.to(device)
            # 因为outputs经过平整，所以labels也要平整来对齐
            label = label.view(-1)

            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()

            output, (state_h, state_c) = model(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})

        # model.eval()
        print('epoch result-----------:', predict(dataset, model, text='窗前明月光，', total_words=48))
        model.train()
        print('saving epoch...', epoch)
        torch.save(model.state_dict(), 'model_data/%s_%s.pth' % ('model', epoch))
    print('Train Finished...........')


def predict(dataset, model, text, total_words=24):
    words = list(text)
    words_len = len(words)
    model.eval()
    # 手动设置第一个词为<START>
    input = torch.Tensor([dataset.word_to_index['<START>']]).view(1, 1).long()
    input = input.to(device)
    hidden = None
    for i in range(total_words):
        if i < words_len:
            w = words[i]
            input = input.data.new([dataset.word_to_index[w]]).view(1, 1)
        else:
            output, hidden = model(input, hidden)
            # 提供的前几个字 直接拼接
            # 预测字的索引
            top_index = output.data[0].topk(1)[1][0].item()
            # 转为字
            w = dataset.index_to_word[top_index]

            # 追加到words中
            words.append(w)
            # 拼接到input中继续传递
            input = input.data.new([top_index]).view(1, 1)

        if w == '<EOP>':
            del words[-1]
            break
    if words[-1] == '，':
        words[-1] = '。'
    words = ''.join(words)
    words = re.sub('。', '。\n', words)
    return words


# 藏头诗
def predict_head(dataset, model, text):
    words = list(text)
    words_len = len(words)
    model.eval()

    pre_word = '。' #也可设置为<START>, 但是会导致第一句出问题
    result = []
    head_index = 0
    # 手动设置第一个词为<START>
    input = torch.Tensor([dataset.word_to_index[pre_word]]).view(1, 1).long()
    input = input.to(device)
    hidden = None

    while True:
        output, hidden = model(input, hidden)

        if (pre_word in ['。', '！', '<START>']):
            w = words[head_index]
            input = input.data.new([dataset.word_to_index[w]]).view(1, 1)
            head_index += 1
            if head_index == words_len:
                break
            result.append(w)
        else:
            # 预测字的索引
            top_index = output.data[0].topk(1)[1][0].item()
            # 转为字
            w = dataset.index_to_word[top_index]
            # 追加到words中
            result.append(w)
            # 拼接到input中继续传递
            input = input.data.new([top_index]).view(1, 1)

        # 记录上一个字
        pre_word = w


        if w == '<EOP>':
            del result[-1]
            break

    result = ''.join(result)
    result = re.sub('。', '。\n', result)

    return result


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=256)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--txt', type=str, default='水花翻照树，')
parser.add_argument('--previous', type=int, default=0)

args = parser.parse_args()

if __name__ == "__main__":

    dataset = Dataset(args)
    model = Model(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 预测模式
    if args.mode == 'test':
        # 加载数据
        model.load_state_dict(torch.load('model_data/trained100.pth', map_location=device))
        print("提供首句（5字），生成的诗如下：")
        print(predict(dataset, model, text=args.txt, total_words=48))
        print("生成的藏头诗如下：")
        print(predict_head(dataset, model, text=args.txt))

    else:
        # 继续之前的训练
        if args.previous != 0:
            pth_file = 'model_data/%s_%s.pth' % ('model', args.previous)
            print('loading previous pth file: ', pth_file)
            model.load_state_dict(torch.load(pth_file, map_location=device))

        train(dataset, model, args)
