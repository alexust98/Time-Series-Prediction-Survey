import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

class RollingWindow():
	def __init__(self, winsize):
		self.buf = torch.empty((winsize, ))
		self.is_init = False
		
	def get(self):
		if not(self.is_init):
			raise ValueError("Buffer is not initialized.")
			return
		return self.buf
		
	def init(self, data):
		assert data.shape == self.buf.shape
		self.buf = data
		
	def get(self):
		return self.buf[-1]
		
	def update(self, value):
		self.buf.roll(-1)
		self.buf[-1] = value

class ConvLSTMCell(nn.Module):
	def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
		super(ConvLSTMCell, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.kernel_size = kernel_size
		self.padding = kernel_size // 2
		self.bias = bias

		self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
							  out_channels=4 * self.hidden_dim,
							  kernel_size=self.kernel_size,
							  padding=self.padding,
							  bias=self.bias)

	def forward(self, input_tensor, cur_state):
		h_cur, c_cur = cur_state
		combined = torch.cat([input_tensor, h_cur], dim=-1)
		combined_conv = self.conv(combined.transpose(1, 2)).transpose(1, 2)

		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=-1)
		i = torch.sigmoid(cc_i)
		f = torch.sigmoid(cc_f)
		o = torch.sigmoid(cc_o)
		g = torch.tanh(cc_g)

		c_next = f * c_cur + i * g
		h_next = o * torch.tanh(c_next)

		return h_next, c_next

	def init_hidden(self, batch_size, input_dim):
		return (torch.zeros(batch_size, input_dim, self.hidden_dim, device=self.conv.weight.device),
				torch.zeros(batch_size, input_dim, self.hidden_dim, device=self.conv.weight.device))
				
class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
								 self.conv2, self.chomp2, self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size, dropout):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
									 padding=(kernel_size-1) * dilation_size, dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)

class ScaledDotProductAttention(nn.Module):
	# Scaled Dot-Product Attention
	def __init__(self, temperature, attn_dropout=0.1):
		super().__init__()
		self.temperature = temperature
		self.dropout = nn.Dropout(attn_dropout)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, q, k, v, mask=None):

		attn = torch.bmm(q, k.transpose(1, 2))
		attn = attn / self.temperature

		if mask is not None:
			attn = attn.masked_fill(mask, -np.inf)

		attn = self.softmax(attn)
		attn = self.dropout(attn)
		output = torch.bmm(attn, v)

		return output, attn

class MultiHeadAttention(nn.Module):
	''' Multi-Head Attention module '''
	def __init__(self, n_head, d_model, d_k, d_v, dropout):
		super().__init__()

		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v

		self.w_qs = nn.Linear(d_model, n_head * d_k)
		self.w_ks = nn.Linear(d_model, n_head * d_k)
		self.w_vs = nn.Linear(d_model, n_head * d_v)
		nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

		self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.2))
		self.layer_norm = nn.LayerNorm(d_model)

		self.fc = nn.Linear(n_head * d_v, d_model)
		nn.init.xavier_normal_(self.fc.weight)

		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, mask=None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

		sz_b, len_q, _ = q.size()
		sz_b, len_k, _ = k.size()
		sz_b, len_v, _ = v.size()

		residual = q

		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

		q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
		k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
		v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
		if mask is not None:
			mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
		output, attn = self.attention(q, k, v, mask=mask)

		output = output.view(n_head, sz_b, len_q, d_v)
		output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

		output = self.dropout(self.fc(output))
		output = self.layer_norm(output + residual)

		return output, attn