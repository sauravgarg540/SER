# !/usr/bin/env python
# coding: utf-8
import torch
import data
from data import Fragment
from alexnetModel import AlexNet
import train
import os
import argparse
# Main program by Ma

test_size=500
valid_size=500
batch_size=50
lr=0.0005
epochs=20
save_dir="/home/zzhang/test/experiment"
early_stop=0.005
snapshot=0
device=0
IfTest=False
IfCuda=True


if __name__ == '__main__':
#load data
	train1, test, valid=data.generatSet(test_size, valid_size)
	trainset=data.dataSet(train1)
	validset=data.dataSet(valid)
	testset=data.dataSet(test)
	#model
	mymodel=AlexNet()
	# if IfTest:
#print('\nLoading model from {}...'.format(snapshot))
#   	mymodel.load_state_dict(torch.load(snapshot))

	if IfCuda:
		torch.cuda.set_device(device)
		mymodel = mymodel.cuda()

	trainloader=torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, drop_last=True, collate_fn=data.collate_fn)
	validloader=torch.utils.data.DataLoader(validset, batch_size=batch_size,shuffle=True, drop_last=True, collate_fn=data.collate_fn)
	testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True, drop_last=True, collate_fn=data.collate_fn)
	train.train(trainloader, validloader, mymodel, lr, epochs, save_dir=save_dir, early_stop=early_stop, save_interval=3, save_best=True, cuda=True, log_interval=1, test_interval=10, batch_size=batch_size)
	train.eval(testloader, mymodel)
	print("Finsh!!")
