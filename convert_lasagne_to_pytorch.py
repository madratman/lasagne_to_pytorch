import lasagne
import torch
import torch.nn as nn
from collections import OrderedDict
from lasagne.layers import DilatedConv2DLayer, PadLayer, InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
import h5py
import numpy as np
import pdb
import colors as col
from models.pytorch import dilated_3 as dilated_3_pytorch
from models.lasagne import dilated_3 as dilated_3_lasagne

def build_pytorch_model():
	return dilated_3_pytorch.build_model()

def build_lasagne_model(height, width):
	return dilated_3_lasagne.build_model()

def lasagne_load_weights(model, weight_file):
	params = lasagne.layers.get_all_params(model['l_out'])
	names = [ par.name for par in params ]
	#print names
	if len(names)!=len(set(names)):
		print len(names), len(set(names))
		raise ValueError('need unique param names')
	with h5py.File(weight_file, 'r') as f:
		for param in params:
			if param.name in f:
				stored_shape = np.asarray(f[param.name].value.shape)
				param_shape = np.asarray(param.get_value().shape)
				# print param_shape, stored_shape
				if not np.all( stored_shape == param_shape ):
					warn_msg = 'shape mismatch:'
					warn_msg += '{} stored:{} new:{}'.format(param.name, stored_shape, param_shape)
					warn_msg += ' skipping'
					print (warn_msg)
				else:
					param.set_value(f[param.name].value)
			else:
				print ('unable to load parameter {} from {}'.format(param.name, weight_file))

def convert_lasagne_to_pytorch(lasagne_weight_file, path_to_save_pytorch_model):
	COLORS = col.Colors() 
	pytorch_model = build_pytorch_model()
	lasagne_model = build_lasagne_model(height=720, width=1280)
	lasagne_load_weights(lasagne_model, lasagne_weight_file)
	pytorch_model_dict = pytorch_model.state_dict()

	lasagne_params = lasagne.layers.get_all_params(lasagne_model['fuse3'])
	lasagne_names = [ par.name for par in lasagne_params ]

	for (lasagne_name, lasagne_param, pytorch_key) in zip(lasagne_names, lasagne_params, pytorch_model_dict):
		pytorch_shape = ()
		for each_dim in range(pytorch_model_dict[pytorch_key].dim()): # this is length of tensor
			pytorch_shape = pytorch_shape +(pytorch_model_dict[pytorch_key].size(each_dim),)
			# print pytorch_key, pytorch_shape
		if lasagne_param.get_value().shape ==  pytorch_shape:
			print "lasagne {:>12}  {:>15}    <===>    pytorch {:>12}  {:>12}".format(lasagne_name, lasagne_param.get_value().shape,\
				 pytorch_key, pytorch_shape)
			pytorch_model_dict[pytorch_key] = torch.from_numpy(lasagne_param.get_value()).float()
		else:
			# last convlayer in pytroch is (2,64,3,3) instead of (64,2,3,3)
			print COLORS.RED, "ERROR at ",  "lasagne_name", lasagne_name, lasagne_param.get_value().shape, \
				" :::: pytorch", pytorch_key, pytorch_shape, COLORS.ENDC
			print COLORS.BLUE, "I am reshaping lasagne weight from {} to {}".\
				format(lasagne_param.get_value().shape, pytorch_shape), COLORS.ENDC 
			lasagne_weight = lasagne_param.get_value()
			pytorch_model_dict[pytorch_key] = torch.from_numpy(lasagne_weight.reshape(pytorch_shape)).float()
		# print lasagne_param.get_value()
		# print pytorch_model_dict[pytorch_key]
		# print "\n\n\n\n\n\n"

	# pdb.set_trace()
	pytorch_model.load_state_dict(pytorch_model_dict) # assign weights to the model's state dict
	torch.save(pytorch_model.state_dict(), path_to_save_pytorch_model)

if __name__=='__main__':
	lasagne_weight_file = 'bla.h5'
	path_to_save_pytorch_model = 'bla.pth'
	convert_lasagne_to_pytorch(lasagne_weight_file, path_to_save_pytorch_model)
