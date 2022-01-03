import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_utils import CustomDataset

from config import PARAMS as params
from utils import init_models, draw

def run_pipeline(paths):
	for d_num, (data, models) in enumerate(init_models(paths, params)):
		train_thresh = int(params["data"]["train_test_split"] * data.shape[0])
		#val_thresh = int(params["data"]["val_test_split"] * data.shape[0])
		train_dataloader = DataLoader(CustomDataset(data[:train_thresh], params), shuffle=True, batch_size = 1)
		#val_dataloader = DataLoader(CustomDataset(data[train_thresh:train_thresh+val_thresh], params, params["models"]["num_steps_to_predict"], params["data"]["train_test_split"]), shuffle=True, batch_size = 1)
		test_dataloader = DataLoader(CustomDataset(data, params, params["models"]["num_steps_to_predict"], params["data"]["train_test_split"]), shuffle=False, batch_size = 1)
		
		for num, model in enumerate(models):
			train_loss = model.fit(dataloader=train_dataloader, params=params)
			#_, val_loss = model.predict(dataloader=val_dataloader, params=params)
			prediction, test_loss = model.predict(dataloader=test_dataloader, params=params)
			#draw(prediction, test_loss, test_dataloader.dataset.data[:, params["data"]["target_idx"]], 0, train_thresh, str(model), paths[d_num])
			yield test_loss
		