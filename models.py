import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF, ConstantKernel, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR as SVRmodel
from model_utils import *
from sklearn.metrics import mean_squared_error as mse
from torch.nn.functional import mse_loss

class DLmodel(nn.Module):
	def train_mode(self, mode=True):
		"""Sets the module in training mode."""	  
		self.training = mode
		for module in self.children():
			module.train(mode)
		return self
			
	def fit(self, dataloader, params, **kwargs):
		self.train_mode(True)
		optimizer = params["optimize"]["optimizer"](self.parameters(), lr=params["optimize"]["lr"])
		for epoch in range(params["optimize"]["num_epochs"]):
			for num, (batch, target) in enumerate(dataloader):
				optimizer.zero_grad()
				pred = self.forward(batch).view(target.shape)
				loss = params["optimize"]["criterion"](pred, target)
				loss.backward()
				optimizer.step()
			if (params["optimize"]["verbose"]):
				print(f"Epoch: {epoch+1} || Loss: {loss.item()}")
		self.train_mode(False)
		return np.sqrt(loss.item())
		
	def predict(self, dataloader, params):
		self.train_mode(False)
		winsize = params["data"]["win_size"]
		result = torch.zeros((len(dataloader) + winsize, ))
		RW = RollingWindow(winsize)
		NSTP = params["models"]["num_steps_to_predict"]
		running_loss = torch.zeros((NSTP, ))
		
		with torch.no_grad():
			for num, (batch, target) in enumerate(dataloader):
				if (num == 0):
					RW.init(batch[0, :winsize, params["data"]["target_idx"]])
					result[:winsize] = batch[0, :winsize, params["data"]["target_idx"]]
				batch[0, -NSTP, params["data"]["target_idx"]] = RW.get()
				for pred_num in range(NSTP):
					pred = self.forward(batch[:, pred_num: winsize + pred_num, :])
					running_loss[pred_num] += mse_loss(pred, target[0, pred_num]).cpu()
					
					if ((winsize + num + pred_num) < len(result)):
						result[winsize + num + pred_num] = pred
					if (pred_num < NSTP - 1):
						batch[0, winsize + pred_num, params["data"]["target_idx"]] = pred
				RW.update(target[:, 0])
		return result, torch.sqrt(running_loss/len(dataloader)).numpy()
		
class MLmodel():
	def fit(self, dataloader, params, **kwargs):
		winsize = params["data"]["win_size"]
		X, Y = np.zeros((len(dataloader), winsize*params["data"]["input_dim"])), np.zeros((len(dataloader), ))
		
		for num, (batch, target) in enumerate(dataloader):
			X[num] = batch[0].view(-1).cpu().numpy()
			Y[num] = target[0].cpu().numpy()
		self.model.fit(X, Y)
		return np.sqrt(mse(self.model.predict(X), Y))
		
	def predict(self, dataloader, params, **kwargs):
		winsize = params["data"]["win_size"]
		result = torch.zeros((len(dataloader) + winsize, ))
		RW = RollingWindow(winsize)
		NSTP = params["models"]["num_steps_to_predict"]
		running_loss = np.zeros((NSTP, ))
		
		for num, (batch, target) in enumerate(dataloader):
			if (num == 0):
				RW.init(batch[0, :winsize, params["data"]["target_idx"]])
				result[:winsize] = batch[0, :winsize, params["data"]["target_idx"]]
			batch[0, -NSTP, params["data"]["target_idx"]] = RW.get()
			for pred_num in range(NSTP):
				pred = torch.from_numpy(self.model.predict(batch[:, pred_num: winsize + pred_num, :].view(1, -1).cpu().numpy()))
				running_loss[pred_num] += mse(pred.cpu().numpy(), target[:, pred_num].cpu().numpy())
				
				if ((winsize + num + pred_num) < len(result)):
					result[winsize + num + pred_num] = pred
				if (pred_num < NSTP - 1):
					batch[0, winsize + pred_num, params["data"]["target_idx"]] = pred
			RW.update(target[:, 0])
		return result, np.sqrt(running_loss/len(dataloader))

class LSTM(DLmodel):
	def __init__(self, params):
		super().__init__()
		self.hidden_dim = params["models"]["hidden_dim"]
		self.lstm = nn.LSTM(params["data"]["input_dim"], params["models"]["hidden_dim"], batch_first=True)

		self.linear = nn.Linear(params["models"]["hidden_dim"], 1)

	def forward(self, sentence):
		lstm_out, _ = self.lstm(sentence)
		out = self.linear(lstm_out[:, -1, :]).squeeze()
		return out
		
	def __str__(self):
		return "LSTM"
	
class GRU(DLmodel):
	def __init__(self, params):
		super().__init__()
		self.hidden_dim = params["models"]["hidden_dim"]
		self.gru = nn.GRU(params["data"]["input_dim"], params["models"]["hidden_dim"], batch_first=True)

		self.linear = nn.Linear(params["models"]["hidden_dim"], 1)

	def forward(self, sentence):
		gru_out, _ = self.gru(sentence)
		out = self.linear(gru_out[:, -1, :]).squeeze()
		return out
		
	def __str__(self):
		return "GRU"

class ConvLSTM(DLmodel):
	def __init__(self, params):
		super(ConvLSTM, self).__init__()

		self.kernel_size = self._extend_for_multilayer(params["models"]["kernel_size"], params["models"]["num_layers"])
		self.hidden_dim = self._extend_for_multilayer(params["models"]["hidden_dim"], params["models"]["num_layers"])

		self.input_dim = params["data"]["input_dim"]
		self.num_layers = params["models"]["num_layers"]
		self.bias = params["models"]["bias"]


		cell_list = []
		for i in range(0, self.num_layers):
			cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

			cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
										  hidden_dim=self.hidden_dim[i],
										  kernel_size=self.kernel_size[i],
										  bias=self.bias))

		self.cell_list = nn.ModuleList(cell_list)
		self.linear = nn.Linear(self.hidden_dim[0], 1)

	def forward(self, input_tensor, hidden_state=None):
		b, winlen, c = input_tensor.size()

		if hidden_state is not None:
			raise NotImplementedError()
		else:
			hidden_state = self._init_hidden(batch_size=b, input_dim=self.kernel_size[0])

		layer_output_list = []
		last_state_list = []

		seq_len = input_tensor.size(1)
		cur_layer_input = input_tensor

		for layer_idx in range(self.num_layers):
			kernel_size = self.kernel_size[layer_idx]
			h, c = hidden_state[layer_idx]
			output_inner = []
			for t in range(seq_len - kernel_size + 1):
				h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t:t+kernel_size, :],
												 cur_state=[h, c])
				output_inner.append(h)

			layer_output = torch.stack(output_inner, dim=1)
			cur_layer_input = layer_output

			layer_output_list.append(layer_output)
			last_state_list.append([h, c])
		out = self.linear(h[:, -1, :]).squeeze()
		return out

	def _init_hidden(self, batch_size, input_dim):
		init_states = []
		for i in range(self.num_layers):
			init_states.append(self.cell_list[i].init_hidden(batch_size, input_dim))
		return init_states

	@staticmethod
	def _extend_for_multilayer(param, num_layers):
		if not isinstance(param, list):
			param = [param] * num_layers
		return param
		
	def __str__(self):
		return "ConvLSTM"
		
class TCN(DLmodel):
	def __init__(self, params):
		super(TCN, self).__init__()
		num_channels = [params["models"]["hidden_dim"]]*params["models"]["num_layers"]
		self.tcn = TemporalConvNet(params["data"]["input_dim"], num_channels, params["models"]["kernel_size"], params["models"]["dropout"])
		self.linear = nn.Linear(num_channels[-1], 1)

	def forward(self, x):
		# x needs to have dimension (N, L, C) in order to be passed into CNN
		output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
		output = self.linear(output[:, -1, :]).squeeze()
		return output
		
	def __str__(self):
		return "TCN"
		
class LSTNet(DLmodel):
	def __init__(self, params):
		super(LSTNet, self).__init__()
		self.conv = nn.Conv1d(params["data"]["input_dim"], params["models"]["hidden_dim"], params["models"]["kernel_size"], padding=params["models"]["kernel_size"]//2, bias=params["models"]["bias"])
		self.gru = nn.GRU(params["models"]["hidden_dim"], params["models"]["hidden_dim"])
		self.dropout = nn.Dropout(p=params["models"]["dropout"])
		self.slf_attn = MultiHeadAttention(params["models"]["n_att_heads"], params["models"]["hidden_dim"], params["models"]["hidden_dim"], params["models"]["hidden_dim"], dropout=params["models"]["dropout"])
		self.linear = nn.Linear(params["models"]["hidden_dim"], 1)

	def forward(self, x):
		x = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
		x = self.dropout(x)

		x, _ = self.gru(x)
		x = self.dropout(x)
		
		x, _ = self.slf_attn(x, x, x)
		res = self.linear(x[:, -1, :]).squeeze()
		
		return res
		
	def __str__(self):
		return "LSTNet"
		
		
class GPR(MLmodel):
	def __init__(self, params):
		k0 = WhiteKernel()
		k1 = ConstantKernel() * ExpSineSquared()
		k2 = ConstantKernel() * RBF()
			
		self.model = GaussianProcessRegressor(
			kernel=k0+k1+k2, 
			n_restarts_optimizer=params["optimize"]["num_epochs"]
			)
		
	def __str__(self):
		return "GPR"
		
class Baseline():
	def __init__(self, params):
		pass
		
	def fit(self, dataloader, params, **kwargs):
		return 0.0
		
	def predict(self, dataloader, params, **kwargs):
		winsize = params["data"]["win_size"]
		result = torch.zeros((len(dataloader) + winsize, ))
		RW = RollingWindow(winsize)
		NSTP = params["models"]["num_steps_to_predict"]
		running_loss = np.zeros((NSTP, ))
		
		for num, (batch, target) in enumerate(dataloader):
			if (num == 0):
				RW.init(batch[0, :winsize, params["data"]["target_idx"]])
				result[:winsize] = batch[0, :winsize, params["data"]["target_idx"]]
			batch[0, -NSTP, params["data"]["target_idx"]] = RW.get()
			for pred_num in range(NSTP):
				pred = torch.from_numpy(batch[:, winsize + pred_num - 1, params["data"]["target_idx"]].view(1, -1).cpu().numpy())
				running_loss[pred_num] += mse(pred.cpu().numpy(), target[:, pred_num].cpu().numpy())
				
				if ((winsize + num + pred_num) < len(result)):
					result[winsize + num + pred_num] = pred
				if (pred_num < NSTP - 1):
					batch[0, winsize + pred_num, params["data"]["target_idx"]] = pred
			RW.update(target[:, 0])
		return result, np.sqrt(running_loss/len(dataloader))
		
	def __str__(self):
		return "Baseline"
		
class SVR(MLmodel):
	def __init__(self, params):
		self.model = SVRmodel(C=params["models"]["C"], epsilon=params["models"]["epsilon"], gamma="scale")
		
	def __str__(self):
		return "SVR"