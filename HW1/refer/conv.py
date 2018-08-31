#!/usr/bin/python
import torch
from torchvision import transforms
#from conv import Conv2D
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import cv2


class Conv2D(object):
	def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
		# init the Conv2D
		self.in_channel = in_channel
		self.o_channel = o_channel
		self.kernel_size = kernel_size
		self.stride = stride
		self.mode = mode
		K1 = torch.FloatTensor([[-1,-1,-1],
							[0,0,0], 
							[1,1,1]])
		K2 = torch.FloatTensor([[-1,0,1],
							[-1,0,1], 
							[-1,0,1]])
		K3 = torch.FloatTensor([[1,1,1],
							[1,1,1], 
							[1,1,1]])
		K4 = torch.FloatTensor([[-1,-1,-1,-1,-1],
							[-1,-1,-1,-1,-1],
							[0,0,0,0,0],
							[1,1,1,1,1],
							[1,1,1,1,1]])
		K5 = torch.FloatTensor([[-1,-1,0,1,1],
							[-1,-1,0,1,1],
							[-1,-1,0,1,1],
							[-1,-1,0,1,1],
							[-1,-1,0,1,1]])
		
		if mode == 'rand':
			self.kernels = [torch.randn(kernel_size, kernel_size) for i in np.arange(self.o_channel)]
		elif mode == 'known':
			if o_channel == 1:
				self.kernels = [K1]
			elif o_channel == 2:
				self.kernels = [K4, K5]
			elif o_channel == 3:
				self.kernels = [K1, K2, K3]
			else:
				raise ValueError('only support O_channel equals to 1,2,3')
		else:
			raise ValueError('mode must be rand or known. Otherwise cannot support in this code')
			
	
	def single_conv(self, img, kernel):
		#print kernel.shape
		#print img.shape
		num_ops = 0
		k_height, k_width = kernel.shape
		#print kernel.shape
		img_height, img_width = img.shape
		
		conv_height = int((img_height - k_height) / self.stride + 1) 
		conv_width = int((img_width - k_width) / self.stride + 1)
		
		#print conv_height, conv_width
		
		conv = torch.FloatTensor(conv_height, conv_width).fill_(0)
		
		
		for h in xrange(conv_height):
			h_pos = h * self.stride
			for w in xrange(conv_width):
				w_pos = w * self.stride
				conv[h][w] = (img[h_pos:h_pos + k_height, w_pos:w_pos + k_width] * kernel).sum()
				#print conv[h][w]
				num_ops += 1
				
		return num_ops, conv	
			
	
	def forward(self, img):
		num_ops = 0
		# convert img to tensors
		img2tensor = transforms.ToTensor()
		img = img2tensor(img)
		conv_imgs = []
		for k in self.kernels:
			
			r_op, r = self.single_conv(img[0], k)
			g_op, g = self.single_conv(img[1], k)
			b_op, b = self.single_conv(img[2], k)
			#print 'r.shape:', r.shape
			# add all the channels together
			r.add(g)
			r.add(b)
			# make the result into a 3D floatTensor
			#conv_img = torch.stack([r])
			conv_imgs.append(r)
		conv_imgs = torch.stack(conv_imgs)
		#print 'conv.shape:', conv_img.shape
		num_ops = r_op + g_op + b_op
		return num_ops, conv_imgs
		
		
