from models import *
import matplotlib.pyplot as plt
import numpy as np

def init_models(paths, params):
	for path in paths:
		data = np.load(path)
		params["data"]["input_dim"] = data.shape[1]
		models = [Baseline(params)]#[LSTM(params).cuda(), GRU(params).cuda(), ConvLSTM(params).cuda(), TCN(params).cuda(), LSTNet(params).cuda(), GPR(params), SVR(params)]
		yield data, models
		
def draw(prediction, loss_vals, gt, draw_last_n, train_thresh, model_name, dataset_name):
		plt.figure(figsize=(15, 9))
		plt.title(f"Inference of {model_name} on {dataset_name} dataset with loss: {str(np.round(loss_vals[0], 5))}", fontsize=20)
		plt.plot(prediction[-draw_last_n:], label='prediction')
		plt.plot(gt[-draw_last_n:], label='gt')
		y_min = min(gt.min(), prediction.min())
		y_max = max(gt.max(), prediction.max())
		if (draw_last_n == 0):
			plt.vlines(x=[train_thresh], ymin=y_min, ymax=y_max, ls='--', label='train')
		elif (train_thresh > data.shape[0] - draw_last_n):
			plt.vlines(x=[train_thresh - (data.shape[0] - draw_last_n)], ymin=y_min, ymax=y_max, ls='--', label='train')
		plt.grid();plt.legend(fontsize=20);plt.show()