from model import PoetryModel
from config import Config
import torch
from dataHandler import load_data
from sample import generate, gen_acrostic


def userTest():
    print("正在初始化......")
    _, char2ix, ix2char = load_data()
    model = PoetryModel(len(ix2char), Config.embedding_dim, Config.hidden_dim)
    model.load_state_dict(torch.load(Config.best_model_path, 'cpu'))
    if Config.use_gpu:
        model.to(torch.device('cuda'))
    print("初始化完成！\n")
    while True:
        print("欢迎使用唐诗生成器，\n"
              "输入1 进入首句生成模式\n"
              "输入2 进入藏头诗生成模式\n")
        mode = int(input())
        if mode == 1:
            print("请输入您想要的诗歌首句，可以是五言或七言")
            start_words = str(input())
            gen_poetry = ''.join(generate(model, start_words, ix2char, char2ix))
            print("生成的诗句如下：%s\n" % (gen_poetry))
        elif mode == 2:
            print("请输入您想要的诗歌藏头部分，不超过16个字，最好是偶数")
            start_words = str(input())
            gen_poetry = ''.join(gen_acrostic(model, start_words, ix2char, char2ix))
            print("生成的诗句如下：\n%s\n" % gen_poetry)


if __name__ == '__main__':
    userTest()
