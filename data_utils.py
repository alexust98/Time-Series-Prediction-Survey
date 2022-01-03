import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
	def __init__(self, data, params, num_steps_to_predict=1, normalize_portion=1.0):
		self.input_dim = data.shape[1]
		self.data = self.normalize(torch.from_numpy(data).float(), normalize_portion)
		self.winsize = params["data"]["win_size"]
		self.target_idx = params["data"]["target_idx"]
		self.num_steps_to_predict = num_steps_to_predict
		
	@property
	def dim(self):
		return self.input_dim

	def __len__(self):
		return len(self.data) - self.winsize - (self.num_steps_to_predict - 1)
		
	@staticmethod
	def normalize(data, normalize_portion, range=(-1, 1), ):
		_data = data[:int(data.shape[0]*normalize_portion)]
		_data = (data-_data.amin(0).unsqueeze(0))/(_data.amax(0).unsqueeze(0)-_data.amin(0).unsqueeze(0))
		return _data*(range[1] - range[0]) + range[0]

	def __getitem__(self, idx):
		return self.data[idx:idx+self.winsize+self.num_steps_to_predict-1, :].cuda(), self.data[idx+self.winsize:idx+self.winsize+self.num_steps_to_predict, self.target_idx].cuda()