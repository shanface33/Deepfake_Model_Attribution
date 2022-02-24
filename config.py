#coding:utf-8
import warnings
warnings.filterwarnings('ignore')

class DefaultConfig(object):
	gpu_device = "1"

	model_name_pre = 'model_name'
	model_path = None  ## the path of the pretrained model
	save_low_bound = 40  ##when the accuracy achieves save_low_bound, the model is saved

	res_plus= 256
	res = 224				
	
	lr = 0.01
	lr_scale = 0.1
	lr_freq_list = [40, 80, 120]

	train_bs = 20 
	test_bs = 20
	test_epoch = 1
	pretrained = True
	pre_path = 'data/resnet50-19c8e357.pth'

	use_gpu = True
	max_epoches = 300


opt = DefaultConfig()
