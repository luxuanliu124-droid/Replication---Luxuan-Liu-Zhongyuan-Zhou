# -*- coding: utf-8 -*-
"""
数据路径配置：1-Linear / 2-GBDT / 3-DNN / 4-ORF 训练与测试使用的输入文件。
默认指向 3-Replication/5-Data/simulated_data.txt；可修改为本地 CSV 路径。
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(_here, "5-Data", "simulated_data.txt")
