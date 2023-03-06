My graduate work
======================================
Documenting my graduate research journey on the subject of electronic   scale readings for s

# requirements
pytorch 1.11.0 
python 3.8.13


#Tesserac
Tesserac是一种基于Transformer的神经网络结构，用于进行自然语言处理（NLP）任务。它由多个Encoder和Decoder模块组成，每个模块包含若干个Transformer层，以实现对输入语言的编码和解码。

数据流动如下：

1.输入语言被经过嵌入层，转换成d维度的语言向量，然后送到第一个Encoder模块。

2.Encoder模块由多个Transformer层堆叠而成，每个Transformer模块接受前一个模块输出向量并生成下一个模块的输入向量。每个Transformer模块包含多头注意力机制、全连接层和残差连接。

3.第一个Encoder模块输出的语言向量在经过多个Encoder模块后，形成了对输入语言的总体编码。

4.编码向量被送入Decoder模块。与Encoder模块相似，Decoder模块也由多个Transformer模块堆叠而成。

5.在每个Transformer模块中，Decoder模块生成注意力掩码，它将被用来限制Encoder模块已经处理过的输入token被重复使用。Decoder还可以生成目标语言的向量表示。

6.在最高层的Decoder模块的输出向量被送到最终的全连接层，用于下一步的评估和预测。

7.整个过程可以用梯度下降法进行训练，以使得Transformer模块的参数具有在给定语言任务上的最佳性能。

总之，Tesserac的网络结构是一个由多个Encoder和Decoder堆叠而成的Transformer模块。它实现了对输入语言的建模和解码，以实现各种自然语言处理任务。



#torch.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

参数：

input：张量，包含预测类别的概率，形状为[N，C]，其中N是batch size，C是类别数。

target：张量，包含真实类别，形状为[N]，每个元素都是0到C-1之间的整数。

weight：可选，张量，表示每个类别的权重，形状为[C]，默认情况下每个类别的权重都是1。

size_average：可选，布尔值，表示是否对每一个batch的损失值求平均。

ignore_index：可选，负整数，表示target中的值为ignore_index的元素会被忽略，不参与损失的计算。

reduce：可选，布尔值，表示是否对每一个batch的损失值求和。

reduction：可选，字符串，表示损失值的计算方式，有两个可选值：‘none’，表示不做任何操作；‘mean’，表示损失值求平均。

示例：

import torch

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
loss = torch.nn.functional.cross_entropy(input, target)
loss.backward()
