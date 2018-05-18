#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/17
"""


def safe_div(x, y):
    """
    安全除法
    :param x: 被除数
    :param y: 除数
    :return: 结果
    """
    x = float(x)
    y = float(y)
    if y == 0.0:
        return 0.0
    else:
        return x / y
