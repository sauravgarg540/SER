#!/usr/bin/env python
# coding: utf-8

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch import optim
from alexnetModel import AlexNet
# Training and evaluation by Ma
def train(train_iter, dev_iter, model, lr, epochs, save_dir, early_stop, save_interval=1, save_best=True, cuda=True, log_interval=1, test_interval=5, batch_size=1):
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, epochs+1):
        print('epoch: {} '.format(epoch))
        for batch in train_iter:
            feature=batch[0]
            target=batch[1]
            if cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            # print(logit)
            #print('logit vector', logit.size())
            #print('target vector', target.size())
            target=torch.max(target,dim=1)[1]
            loss = F.cross_entropy(logit, target)

            loss.backward()
            optimizer.step()
            print(model.parameters())

            steps += 1
            if steps % log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data,
                                                                             accuracy,
                                                                             corrects,
                                                                             batch_size))
        if epoch % test_interval == 0:
            dev_acc = eval(dev_iter, model)
            if dev_acc > best_acc:
                best_acc = dev_acc
                last_step = steps
                if save_best:
                    save(model, save_dir, 'best', steps)
            else:
                if steps - last_step >= early_stop:
                    print('early stop by {} steps.'.format(early_stop))
        elif epoch % save_interval == 0:
            save(model, save_dir, 'snapshot', steps)

def eval(data_iter, model, cuda=True):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature=batch[0]
        target=batch[1]
        target=torch.max(target,dim=1)[1]
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    f=open("/home/zzhang/test/experiment/result.txt","a")
    f.write('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    f.close()
    return accuracy




def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
