#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/17
"""
import os
import random

import mxnet as mx
import tensorflow as tf
import numpy as np

from mxnet import autograd, gluon
from mxnet.gluon.data import DataLoader, dataset
from mxnet.gluon.data.vision import MNIST
from mxnet.gluon.nn import Sequential, Dense
from tensorboard.plugins import projector

from root_dir import ROOT_DIR
from utils import safe_div


class TripletDataset(dataset.Dataset):
    def __init__(self, rd, rl, transform=None):
        self.__rd = rd  # 原始数据
        self.__rl = rl  # 原始标签
        self._data = None
        self._label = None
        self._transform = transform
        self._get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx], self._label[idx])
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._label)

    def _get_data(self):
        label_list = np.unique(self.__rl)
        digit_indices = [np.where(self.__rl == i)[0] for i in label_list]
        tl_pairs = self.create_pairs(self.__rd, digit_indices, len(label_list))
        self._data = tl_pairs
        self._label = np.ones(tl_pairs.shape[0])

    @staticmethod
    def create_pairs(x, digit_indices, num_classes):
        x = x.asnumpy()  # 转换数据格式
        pairs = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
        for d in range(num_classes):
            for i in range(n):
                np.random.shuffle(digit_indices[d])
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z3 = digit_indices[dn][i]
                pairs += [[x[z1], x[z2], x[z3]]]
        return np.asarray(pairs)


def evaluate_net(model, test_data, ctx):
    triplet_loss = gluon.loss.TripletLoss(margin=0)
    sum_correct = 0
    sum_all = 0
    rate = 0.0
    for i, (data, _) in enumerate(test_data):
        data = data.as_in_context(ctx)

        anc_ins, pos_ins, neg_ins = data[:, 0], data[:, 1], data[:, 2]
        inter1 = model(anc_ins)  # 训练的时候组合
        inter2 = model(pos_ins)
        inter3 = model(neg_ins)
        loss = triplet_loss(inter1, inter2, inter3)

        loss = loss.asnumpy()
        n_all = loss.shape[0]
        n_correct = np.sum(np.where(loss == 0, 1, 0))

        sum_correct += n_correct
        sum_all += n_all
        rate = safe_div(sum_correct, sum_all)

    print('准确率: %.4f (%s / %s)' % (rate, sum_correct, sum_all))
    return rate


def tb_projector(X_test, y_test, log_dir):
    """
    TB的映射器
    :param X_test: 数据
    :param y_test: 标签, 数值型
    :param log_dir: 文件夹
    :return: 写入日志
    """
    print "展示数据: %s" % str(X_test.shape)
    print "展示标签: %s" % str(y_test.shape)
    print "日志目录: %s" % str(log_dir)

    metadata = os.path.join(log_dir, 'metadata.tsv')

    images = tf.Variable(X_test)

    # 把标签写入metadata
    with open(metadata, 'w') as metadata_file:
        for row in y_test:
            metadata_file.write('%d\n' % row)

    with tf.Session() as sess:
        saver = tf.train.Saver([images])  # 把数据存储为矩阵

        sess.run(images.initializer)  # 图像初始化
        saver.save(sess, os.path.join(log_dir, 'images.ckpt'))  # 图像存储于images.ckpt

        config = projector.ProjectorConfig()  # 配置
        # One can add multiple embeddings.
        embedding = config.embeddings.add()  # 嵌入向量添加
        embedding.tensor_name = images.name  # Tensor名称
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = metadata  # Metadata的路径
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)  # 可视化嵌入向量


def main():
    ctx = mx.cpu()
    batch_size = 1024
    random.seed(47)

    mnist_train = MNIST(train=True)  # 加载训练
    tr_data = mnist_train._data.reshape((-1, 28 * 28))  # 数据
    tr_label = mnist_train._label  # 标签

    mnist_test = MNIST(train=False)  # 加载测试
    te_data = mnist_test._data.reshape((-1, 28 * 28))  # 数据
    te_label = mnist_test._label  # 标签

    def transform(data_, label_):
        return data_.astype(np.float32) / 255., label_.astype(np.float32)

    train_data = DataLoader(
        TripletDataset(rd=tr_data, rl=tr_label, transform=transform),
        batch_size, shuffle=True)

    test_data = DataLoader(
        TripletDataset(rd=te_data, rl=te_label, transform=transform),
        batch_size, shuffle=True)

    base_net = Sequential()
    with base_net.name_scope():
        base_net.add(Dense(256, activation='relu'))
        base_net.add(Dense(128, activation='relu'))

    base_net.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)

    triplet_loss = gluon.loss.TripletLoss()  # TripletLoss损失函数
    trainer_triplet = gluon.Trainer(base_net.collect_params(), 'sgd', {'learning_rate': 0.05})

    for epoch in range(10):
        curr_loss = 0.0
        for i, (data, _) in enumerate(train_data):
            data = data.as_in_context(ctx)
            anc_ins, pos_ins, neg_ins = data[:, 0], data[:, 1], data[:, 2]
            with autograd.record():
                inter1 = base_net(anc_ins)
                inter2 = base_net(pos_ins)
                inter3 = base_net(neg_ins)
                loss = triplet_loss(inter1, inter2, inter3)  # Triplet Loss
            loss.backward()
            trainer_triplet.step(batch_size)
            curr_loss = mx.nd.mean(loss).asscalar()
            # print('Epoch: %s, Batch: %s, Triplet Loss: %s' % (epoch, i, curr_loss))
        print('Epoch: %s, Triplet Loss: %s' % (epoch, curr_loss))
        evaluate_net(base_net, test_data, ctx=ctx)

    # 数据可视化
    te_data, te_label = transform(te_data, te_label)
    tb_projector(te_data.asnumpy(), te_label, os.path.join(ROOT_DIR, 'logs', 'origin'))
    te_res = base_net(te_data)
    tb_projector(te_res.asnumpy(), te_label, os.path.join(ROOT_DIR, 'logs', 'triplet'))


if __name__ == '__main__':
    main()
