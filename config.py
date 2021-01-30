class Config(object):
    num_layers = 3  # LSTM层数
    data_path = 'data/tang.txt'  # 诗歌的文本文件存放路径
    processed_data_path = 'data/tang.npz'  # 预处理好的二进制文件
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 50
    batch_size = 16
    poetry_max_len = 130  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 200  # 多少个batch可视化一次
    max_gen_len = 130  # 生成诗歌最长长度
    best_model_path = "checkpoints/*"  # 最佳模型
    model_root = "checkpoints/" # 模型保存路径
    embedding_dim = 256
    hidden_dim = 512
