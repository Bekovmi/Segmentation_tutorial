import numpy as np
import cv2
import torch.nn as nn
import torch 
import os
from albumentations.pytorch import ToTensor
from albumentations import (Compose, Normalize)
class CarPredict:
	def __init__(self, model_path,image_size):
		self.transform_pipeline = Compose([Normalize(),ToTensor()])
		self.m = lambda x: 1/(1 + np.exp(-x))
		self.model = torch.jit.load(model_path).cuda()
		device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
        self.image_size = image_size
	def predict(self, image,threshold):
		image = cv2.resize(image,(image_size,image_size))
		augmented = self.augmentation(image=image)
		inputs = augmented['image']
		output = self.m(self.model(inputs)[0][0].cpu().numpy())
        inputs = inputs.to(self.device)	
		probability = (output > threshold).astype(np.uint8)
		return probability
