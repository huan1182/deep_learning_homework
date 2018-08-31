#!/usr/bin/python

import torch
from torchvision import transforms
from conv import Conv2D
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import cv2
import time

def partA(img_name, i_channel, o_channel, kernel_size, stride, mode):
	img = cv2.imread(img_name)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	conv2d = Conv2D(i_channel, o_channel, kernel_size, stride, mode)
	ops, conv_img = conv2d.forward(img)
	# rescale the conv_img from tensor to imgs
	channel, height, width = conv_img.shape
	print channel
	for c in xrange(channel):
		res = conv_img[c].numpy()
		min_p = np.min(res)
		max_p = np.max(res)
		res = np.uint8((res - min_p) / (max_p - min_p) * 255)
		# res_trees_1_1.png
		cv2.imwrite('res_'+img_name[:-4] + '_ ' + str(o_channel) + '_' + str(c)+ '.png', res)
	
def partB(img_name):
	# 3,2^i, 3, 5, rand
	img = cv2.imread(img_name)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	times = []
	for i in xrange(0, 11):
		print i, ' in part B.'
		s_time = time.time()
		o_channel = 2 ** i
		conv2d = Conv2D(3, o_channel, 3, 1, 'rand')
		ops, conv_img = conv2d.forward(img)
		e_time = time.time()
		times.append(e_time - s_time)
		
	plt.xlabel('log(o_channels)')
	plt.ylabel('time')
	plt.title("partB time--o_channel graphs")
	plt.plot([i for i in xrange(11)], times, marker = 'x', color = 'red', label=img_name)
	plt.legend()
	plt.savefig('partB_'+img_name)
	plt.cla()
	
def partC(img_name):
	kernel_sizes = [3,5,7,9,11]
	ops = []
	img = cv2.imread(img_name)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	for ks in kernel_sizes:
		print ks, 'in part C'
		conv2d = Conv2D(3, 2, ks, 1, 'rand')
		op, conv_img = conv2d.forward(img)
		print op
		ops.append(op)
	
	plt.xlabel('kernel size')
	plt.ylabel('the number of ops')
	plt.title('partC number of ops--kernel size')
	plt.plot([i for i in xrange(3, 12, 2)], ops, marker = 'x', color = 'red', label=img_name)
	plt.legend()
	plt.savefig('partC_'+img_name)
	plt.cla()


def main():	
	'''
	# the following is for partA
	partA('trees.jpg', 3, 1, 3, 1, 'known')
	partA('trees.jpg', 3, 2, 5, 1, 'known')
	partA('trees.jpg', 3, 3, 3, 2, 'known')
	partA('mountain.jpg', 3, 1, 3, 1, 'known')
	partA('mountain.jpg', 3, 2, 5, 1, 'known')
	partA('mountain.jpg', 3, 3, 3, 2, 'known')
	'''
	partB('trees.jpg')
	partB('mountain.jpg')
	#partC('trees.jpg')
	#partC('mountain.jpg')
	
		
if __name__ == '__main__':
	main()