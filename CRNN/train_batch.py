import torch
import torch.nn as nn


def train_batch(model, train_iter, optimizer, criterion, device, converter):
    acc_num = 0
    images, labels = train_iter.next()
    y = labels

    images = images.to(device)
    batch_size = images.size(0)
    text, length = converter.encode(labels)
    #print(text.shape)
    text=text.to(device)
    #print('text',text.device)
    #preds = model(images, text) 
    preds = model(images) 
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    print(preds.shape, text.shape, preds_size.shape, length.shape)
    
    # preds = model(images, text[:, :-1])  # align with Attention.forward
    # target = text[:, 1:]  # without [GO] Symbol
    # #print(target)
    # loss = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

    loss = criterion(preds, text, preds_size, length)

    y_hat = nn.functional.softmax(preds, 2).argmax(2).view(preds.size(0), -1)
    y_hat = torch.transpose(y_hat, 1, 0)
    y_hat = [converter.decode(i, torch.IntTensor([y_hat.size(1)])) for i in y_hat]

    for txt, target in zip(y, y_hat):
        if txt == target:
            acc_num += 1

    model.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), batch_size, 