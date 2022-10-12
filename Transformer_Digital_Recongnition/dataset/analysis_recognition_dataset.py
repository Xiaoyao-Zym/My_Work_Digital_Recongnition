# -*- encoding: utf-8 -*-
import cv2

import os
class Analysis_Recognition_Dataset():

    def __init__(self, char_data_dir, base_data_dir):
        # 训练数据集和验证数据集所在路径
        self.train_img_dir = os.path.join(base_data_dir, 'train')
        self.valid_img_dir = os.path.join(base_data_dir, 'val')
        # 训练集和验证集标签文件路径
        self.train_lbl_path = os.path.join(char_data_dir, 'train.txt')
        self.valid_lbl_path = os.path.join(char_data_dir, 'val.txt')
        # 中间文件存储路径，存储标签字符与其id的映射关系
        self.lbl2id_map_path = os.path.join(char_data_dir, 'figure.txt')

    def statistics_label_cnt(self, lbl_path, lbl_cnt_map):
        """
        统计标签文件中都包含哪些label以及各自出现的次数
        """
        with open(lbl_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                items = line.rstrip().split(' ')
                # img_name = items[0]
                lbl_str = items[1]  # 提取标签并去除label中的双引号""
                for lbl in lbl_str:
                    if lbl not in lbl_cnt_map.keys():
                        lbl_cnt_map[lbl] = 1
                    else:
                        lbl_cnt_map[lbl] += 1

    def statistics_max_len_label(self, lbl_path):
        """
        统计标签文件中最长的label所包含的字符数
        lbl_path: txt标签文件路径
        """
        max_len = -1
        with open(lbl_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                items = line.rstrip().split(' ')
                # img_name = items[0]  # 提取图像名称
                lbl_str = items[1]  # 提取标签，并除掉标签中的引号""
                lbl_len = len(lbl_str)
                max_len = max_len if max_len > lbl_len else lbl_len
        return max_len

    def load_lbl2id_map(self):
        """
        读取 字符-id 映射关系记录的txt文件，并返回 lbl->id 和 id->lbl 映射字典
        lbl2id_map_path : 字符-id 映射关系记录的txt文件路径
        """
        lbl2id_map = dict()
        id2lbl_map = dict()
        with open(self.lbl2id_map_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                items = line.split('\t')
                label = items[0]
                cur_id = int(items[1])
                lbl2id_map[label] = cur_id
                id2lbl_map[cur_id] = label
        return lbl2id_map, id2lbl_map

    def label_to_id(self, lbl_cnt_map):
        # 构造 label - id 之间的映射
        print("\n\n构造 label - id 之间的映射")
        lbl2id_map = dict()
        # 初始化三个个特殊字符
        lbl2id_map['☯'] = 0  # padding标识符
        lbl2id_map['■'] = 1  # 句子起始符
        lbl2id_map['□'] = 2  # 句子结束符
        # 生成其余label的id映射关系
        cur_id = 3
        for lbl in lbl_cnt_map.keys():
            lbl2id_map[lbl] = cur_id
            cur_id += 1

        # 保存 label - id 之间的映射 到txt文件
        with open(self.lbl2id_map_path, 'w',
                  encoding='utf-8') as writer:  # 参数encoding是可选项，部分设备并未默认为utf-8
            for lbl in lbl2id_map.keys():
                cur_id = lbl2id_map[lbl]
                print(lbl, cur_id)
                line = lbl + '\t' + str(cur_id) + '\n'
                writer.write(line)

    def analysis_image_shape(self):
        # 分析数据集图片尺寸
        print("\n\n 分析数据集图片尺寸")
        min_h = 1e10
        min_w = 1e10
        max_h = -1
        max_w = -1
        min_ratio = 1e10
        max_ratio = 0
        # 遍历数据集计算尺寸信息
        for img_name in os.listdir(self.train_img_dir):
            img_path = os.path.join(self.train_img_dir, img_name)
            img = cv2.imread(img_path)  # 读取图片
            h, w = img.shape[:2]  # 提取图像高宽信息
            ratio = w / h  # 宽高比
            min_h = min_h if min_h <= h else h  # 最小图片高度
            max_h = max_h if max_h >= h else h  # 最大图片高度
            min_w = min_w if min_w <= w else w  # 最小图片宽度
            max_w = max_w if max_w >= w else w  # 最大图片宽度
            min_ratio = min_ratio if min_ratio <= ratio else ratio  # 最小宽高比
            max_ratio = max_ratio if max_ratio >= ratio else ratio  # 最大宽高比
        # 输出信息
        print("min_h", min_h)
        print("max_h", max_h)
        print("min_w", min_w)
        print("max_w", max_w)
        print("min_ratio", min_ratio)
        print("max_ratio", max_ratio)

    def analysis_recognition_dataset(self):
        # 统计数据集中出现的所有的label中包含字符最多的有多少字符
        max_label_len = max(self.statistics_max_len_label(self.train_lbl_path),
                            self.statistics_max_len_label(
                                self.valid_lbl_path))  # 全数据集最长label
        print(f"数据集中包含字符最多的label长度为{max_label_len}")
        # 统计数据集中出现的所有的符号
        lbl_cnt_map = dict()  # 用于存储字符出现次数的字典
        self.statistics_label_cnt(self.train_lbl_path,
                                  lbl_cnt_map)  # 训练集中字符出现次数统计
        print("训练集中出现的label")
        print(lbl_cnt_map)
        self.statistics_label_cnt(self.valid_lbl_path,
                                  lbl_cnt_map)  # 训练集和验证集中字符出现次数统计
        print("训练集+验证集中出现的label")
        print(lbl_cnt_map)
        self.label_to_id(lbl_cnt_map)
        self.analysis_image_shape()


if __name__ == "__main__":
    # 数据集根目录
    base_data_dir = '../../data/'
    char_data_dir = '../labels/'
    dataset = Analysis_Recognition_Dataset(char_data_dir, base_data_dir)
    dataset.analysis_recognition_dataset()
    