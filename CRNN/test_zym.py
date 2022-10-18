from log.loss import loss
import time
if __name__ == '__main__':
    isTrue=True
    # opt = parse_opt()
    # train(opt)
    # str_time=time.strftime('%Y-%m-%d', time.localtime())
    data_path="./log/train_record/2022-10-18.csv"
    loss(isTrue ,data_path)