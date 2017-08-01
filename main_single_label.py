from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from libs.dataset import kgforest
from libs.preprocessing import prepro
from libs.utils import utils
from sklearn.metrics import accuracy_score, fbeta_score
import pdb

parser = argparse.ArgumentParser(description='PyTorch Kgforest Training')
parser.add_argument('--train_data', type=str,
                    help='path to train dataset')
parser.add_argument('--test', type=bool, default=False,
                    help='path to train dataset')
parser.add_argument('--resume', type=str, default=None,
                    help='path to latest checkpoint')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--test_data', type=str, default='',
                    help='path to test dataset')
parser.add_argument('--train_csv', type=str,
                    help='path to train label csv')
parser.add_argument('--test_csv', type=str, default='',
                    help='path to test label csv')
parser.add_argument('--label_map_txt', type=str,
                    help='path to label map')
parser.add_argument('--out_dir', type=str,
                    help='path to save submission csv')
parser.add_argument('--epochs', type=int, default=25,
                    help='number of total epochs to run')
parser.add_argument('--lr', type=float, default=0.01,
                    help='base learning rate')

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    p2 = (p > x).astype(np.int)
    score = fbeta_score(y, p2, beta=2)
    return score

  x = 0.21
  best_score = 0
  best_i = 0
  for i in range(1,resolution):
	i /= resolution
	x = i
	score = mf(x)
	if score > best_score:
	  best_i = i
	  best_score = score
	x = best_i
  if verbose:
    print(best_i, best_score)

  return x

def label_smoothing(targets):
	smooth_value = 0.1
	return (targets*(1-smooth_value)+smooth_value/2).astype(np.float32)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = lr * (0.1 ** (epoch // 10))
    
    for i  in range(len(optimizer.param_groups)):
        if i==0:
            optimizer.param_groups[i]['lr'] = args.lr * (0.1 ** (epoch // 10))
            # print('fc layer lr:{:.6f}'.format(optimizer.param_groups[i]['lr']))
        else:
            optimizer.param_groups[i]['lr'] = args.lr/10 * (0.1 ** (epoch // 10))
            # print('conv layer lr:{:.8f}'.format(optimizer.param_groups[i]['lr']))

def save_fn(state, filename='./ckpt/res18_s6_ls_sgd_data.pth.tar'):
    torch.save(state, filename)

def train(data_loader, eval_loader, model, epochs, optimizer):
	"""train the model"""

	start = time.time()
	best_acc = 0
	criterion = torch.nn.BCELoss()

	for epoch in range(args.start_epoch, epochs):
		adjust_learning_rate(optimizer, epoch)

		model.train()

		for idx, (imgs, targets) in enumerate(data_loader):

			imgs, targets = Variable(imgs).cuda(), Variable(targets).cuda()
			optimizer.zero_grad()
			output = model(imgs)
			output = F.sigmoid(output)
			# print(type(output))
			size = output.size()[0]
			output = output.view([size,])
			# print(output.size(), targets.size())
			loss = criterion(output, targets)
			loss.backward()
			optimizer.step()

			if idx%10 == 0:
				print('Train Epoch:{}\tStep:{}/{}\tLoss:{:.6f}\tfscore:{:.6f}\tSec:{:.5f}'.format(
						epoch, idx, \
						len(data_loader.dataset)//len(imgs),
						loss.data[0], \
						fbeta_score(np.round(targets.data.cpu().numpy()).astype(np.int), \
									(output.data.cpu().numpy()>0.21).astype(np.int), \
									beta=2),
						time.time()-start))

				start = time.time()
		cur_acc = evalution_in_train(eval_loader, model)
		print('Cur_fscore:{:.6f}\tBest_fscore:{:.6f}'.format(cur_acc, best_acc))
		if cur_acc > best_acc:
			save_fn({
				'epoch': epoch + 1,
	            'state_dict': model.state_dict(),
	            'optimizer' : optimizer.state_dict(),
				})
			best_acc = cur_acc
		if epoch == epochs-1:
			save_fn({'epoch': epoch+1,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),}, filename='./ckpt/final_s.pth.tar')
def evalution_in_train(data_loader, model):
	"""eval the model and return the best thresholds for pred"""
	data_pred, data_target = [], []

	model.eval()

	for idx, (imgs, targets) in enumerate(data_loader):

		imgs = Variable(imgs, volatile=True).cuda()
		output = model(imgs)
		pred = F.sigmoid(output).data.cpu().numpy()
		targets = targets.numpy()
		data_pred.append(pred)
		data_target.append(targets)
	
	targets = np.concatenate(data_target, axis=0)
	pred = np.concatenate(data_pred, axis=0)
	return 	fbeta_score(targets.astype(np.int), \
						(pred>0.21).astype(np.int), beta=2)

def evalution(data_loader, model):
	"""eval the model and return the best thresholds for pred"""
	data_pred, data_target = [], []

	model.eval()

	for idx, (imgs, targets) in enumerate(data_loader):

		imgs = Variable(imgs, volatile=True).cuda()
		output = model(imgs)
		pred = F.sigmoid(output).data.cpu().numpy()
		targets = np.round(targets.numpy()).astype(np.int)
		data_pred.append(pred)
		data_target.append(targets)
	
	targets = np.concatenate(data_target, axis=0)
	pred = np.concatenate(data_pred, axis=0).reshape(targets.shape)

	# pdb.set_trace()
	predict = (pred>0.21).astype(np.int)
	res = np.sum((targets == predict).astype(np.int))/targets.shape[0]
	print(res)

	return optimise_f2_thresholds(targets, pred)

def test(data_loader, model, thres):
	"""test the model and write to csv file"""
	data_pred = []
	model.eval()

	for idx, imgs in enumerate(data_loader):

		imgs = Variable(imgs, volatile=True).cuda()
		output = model(imgs)
		pred = F.sigmoid(output).data.cpu().numpy()
		data_pred.append(pred)

	data_pred = np.concatenate(data_pred, axis=0)
	label_map, inv_label_map = utils.load_label_map(args.label_map_txt)
	labels = np.array([label_map[i] for i in range(data_loader.dataset.num_classes)])
	predict = ['n' for i in range(data_pred.shape[0])]
	for i in range(data_pred.shape[0]):
		if data_pred[i,0]>thres:
			predict[i] = label_map[6]

	# pdb.set_trace()

	image_names = pd.read_csv(args.test_csv)['image_name']
	sub = pd.DataFrame()
	sub['image_name'] = image_names
	sub['tags'] = predict
	sub.to_csv(os.path.join(args.out_dir, 's_6.csv'), index=False)

def main():
	global args 
	args = parser.parse_args()

	# import pdb
	# pdb.set_trace()
	data_transform = {

				'train':transforms.Compose(
					[
						# transforms.Lambda(lambda x:prepro.random_scale(x, max_size=356)),
						# transforms.Scale(234),
						# transforms.RandomCrop(224), 
						transforms.Scale(224),
						transforms.RandomHorizontalFlip(), 
						transforms.Lambda(lambda x:prepro.random_vflip(x)), 
						transforms.Lambda(lambda x:prepro.random_transpose(x)), 
						transforms.Lambda(lambda x:prepro.random_rotate(x)),
						#transforms.Lambda(lambda x:prepro.random_brightness(x, 0.8)),
						#transforms.Lambda(lambda x:prepro.random_contrast(x, 0.8)), 
						transforms.ToTensor(), 
						transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
					]),
				'val':transforms.Compose(
					[
						transforms.Scale(224),
						# transforms.CenterCrop(224), 
						transforms.ToTensor(), 
						transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
					])
				}
	train_dataset = kgforest.kgforest_single(
						args.train_data, \
						args.train_csv, \
						'train', \
						transform=data_transform['train'],
						target_transform=label_smoothing,
						label_map_txt=args.label_map_txt,
							)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, \
						num_workers=4, pin_memory=True, drop_last=True)

	eval_dataset = kgforest.kgforest_single(
						args.train_data, \
						args.train_csv, \
						'eval', \
						transform=data_transform['val'],
						label_map_txt=args.label_map_txt,
							)

	eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False, \
						num_workers=4, pin_memory=True, drop_last=True)

	if args.test:
		test_dataset = kgforest.kgforest_single(
							args.test_data, \
							args.test_csv, \
							'test', \
							transform=data_transform['val'],
							label_map_txt=args.label_map_txt,
								)

		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, \
							num_workers=4, pin_memory=True)

	num_classes = train_dataset.num_classes
	net = models.resnet18(pretrained=True)
	fc_infeatures = net.fc.in_features
	net.fc = nn.Linear(fc_infeatures, num_classes)
	net.cuda()

	# optimizer = optim.Adam([{'params': net.fc.parameters(), \
	# 						'lr': args.lr}], lr=args.lr/10)
	#optimizer = optim.Adam(net.parameters(), lr=args.lr)#, weight_decay=0.00001)
	fc_optim_config = [{'params':net.fc.parameters(), 'lr':args.lr}]
	conv_optim_config = [{'params':p.parameters()} for p in list(net.children())[:-1]]
	optim_config = fc_optim_config + conv_optim_config	
	optimizer = optim.SGD(optim_config, lr=args.lr/10, momentum=0.9)
	if args.resume:
		if os.path.isfile(args.resume):
			print("load checkpoint from '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			net.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
		else:
			print("no checkpoint found at '{}'".format(args.resume))

	#train(train_loader, eval_loader, net, epochs=args.epochs, optimizer=optimizer)
	best_thres = evalution(eval_loader, net)
	#best_thres = evalution(train_loader, net)
	if args.test:
		test(test_loader, net, best_thres)

if __name__ == '__main__':
	main()

