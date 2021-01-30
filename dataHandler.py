#!/usr/bin/env python

import os
import json
import numpy as np
from config import Config


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def load_data():
    # 如果存在处理过的文件，那直接加载，不用再处理
    if os.path.exists(Config.processed_data_path):
        print("loading processed data...")
        data = np.load(Config.processed_data_path, allow_pickle=True)
        pad_data, char2ix, id2char = data['data'], data['char2ix'].item(), data['ix2char'].item()
        return pad_data, char2ix, id2char

    # 1.加载原始数据
    print("Processed raw data...")
    data = open(Config.data_path,encoding='utf-8').read().split('\n')

    # 2.构建词典
    chars = {c for line in data for c in line}
    char2ix = {char: ix for ix, char in enumerate(chars)}
    char2ix['<START>'] = len(char2ix)
    char2ix['<END>'] = len(char2ix)
    char2ix['<PAD>'] = len(char2ix)

    ix2char = {ix: char for char, ix in list(char2ix.items())}

    # 3.处理样本
    # 3.1 每首诗加上首位符号
    for i in range(0, len(data)):
        data[i] = ['<START>'] + list(data[i]) + ['<END>']

    # 3.2 文字转ix
    data_id = [[char2ix[w] for w in line] for line in data]

    # 3.3 补全既定长度
    pad_data = pad_sequences(data_id,
                             maxlen=Config.poetry_max_len,
                             padding='pre',
                             truncating='post',
                             value=len(char2ix) - 1)

    # 3.4 保存于返回
    np.savez_compressed(Config.processed_data_path,
                        data=pad_data,
                        char2ix=char2ix,
                        ix2char=ix2char)
    print("Saved processed data in %s" % Config.processed_data_path)
    return pad_data, char2ix, ix2char

