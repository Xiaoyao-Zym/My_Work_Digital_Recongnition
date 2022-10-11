My graduate work
======================================
Documenting my graduate research journey on the subject of electronic   scale readings for s

# requirements
pytorch 1.11.0 
python 3.8.13


# Test
pretrained model coming soon


# Train 
1. Here i choose a small dataset from [Synthetic_Chinese_String_Dataset](https://github.com/chenjun2hao/caffe_ocr), about 270000+ images for training, 20000 images for testing.
download the image data from [Baidu](https://pan.baidu.com/s/1hIurFJ73XbzL-QG4V-oe0w)
2. the train_list.txt and test_list.txt are created as the follow form:
```
# path/to/image_name.jpg label
path/AttentionData/50843500_2726670787.jpg 情笼罩在他们满是沧桑
path/AttentionData/57724421_3902051606.jpg 心态的松弛决定了比赛
path/AttentionData/52041437_3766953320.jpg 虾的鲜美自是不可待言
```
3. change the **trainlist** and **vallist** parameter in train.py, and start train
```bash
cd Attention_ocr.pytorch
python train.py --trainlist ./data/ch_train.txt --vallist ./data/ch_test.txt
```
then you can see in the terminel as follow:
![attentionocr](./test_img/md_img/attentionV2.png)
there uses the decoderV2 model for decoder.


# The previous version  

**_git checkout AttentionOcrV1_**


# Reference
1. [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
2. [Attention-OCR](https://github.com/da03/Attention-OCR)
3. [Seq2Seq-PyTorch](https://github.com/MaximumEntropy/Seq2Seq-PyTorch)
4. [caffe_ocr](https://github.com/senlinuc/caffe_ocr)


# TO DO
- [ ] change LSTM to Conv1D, it can greatly accelerate the inference
- [ ] change the cnn bone model with inception net, densenet
- [ ] realize the decoder with transformer model
