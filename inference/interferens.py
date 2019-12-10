import numpy as np
import cv2
import torch.nn as nn
import torch 
import os
from ..src/transforms import (pre_transforms, post_transforms, Compose)
class CarPredict:
	def __init__(self, model_path,image_size):
		self.transform_pipeline =Compose([pre_transforms(image_size=image_size),post_transforms()])
		self.m = nn.Sigmoid()
		self.model = torch.jit.load(model_path).cuda()
		device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
        self.image_size = image_size
	def predict(self, image,threshold):
		augmented = self.augmentation(image=image)
		inputs = augmented['image']
        inputs = inputs.to(self.device)	
		output = self.m(self.model(inputs)[0][0]).cpu().numpy()
		probability = (output > threshold).astype(np.uint8)
		return probability
