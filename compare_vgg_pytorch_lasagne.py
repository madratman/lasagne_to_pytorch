import lasagne
import cPickle as pkl
import pdb
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

def build_model():
	net = {}
	net['input'] = InputLayer((None, 3, 224, 224))
	net['conv1_1'] = ConvLayer(
		net['input'], 64, 3, pad=1, flip_filters=False)
	net['conv1_2'] = ConvLayer(
		net['conv1_1'], 64, 3, pad=1, flip_filters=False)
	net['pool1'] = PoolLayer(net['conv1_2'], 2)
	net['conv2_1'] = ConvLayer(
		net['pool1'], 128, 3, pad=1, flip_filters=False)
	net['conv2_2'] = ConvLayer(
		net['conv2_1'], 128, 3, pad=1, flip_filters=False)
	net['pool2'] = PoolLayer(net['conv2_2'], 2)
	net['conv3_1'] = ConvLayer(
		net['pool2'], 256, 3, pad=1, flip_filters=False)
	net['conv3_2'] = ConvLayer(
		net['conv3_1'], 256, 3, pad=1, flip_filters=False)
	net['conv3_3'] = ConvLayer(
		net['conv3_2'], 256, 3, pad=1, flip_filters=False)
	net['pool3'] = PoolLayer(net['conv3_3'], 2)
	net['conv4_1'] = ConvLayer(
		net['pool3'], 512, 3, pad=1, flip_filters=False)
	net['conv4_2'] = ConvLayer(
		net['conv4_1'], 512, 3, pad=1, flip_filters=False)
	net['conv4_3'] = ConvLayer(
		net['conv4_2'], 512, 3, pad=1, flip_filters=False)
	net['pool4'] = PoolLayer(net['conv4_3'], 2)
	net['conv5_1'] = ConvLayer(
		net['pool4'], 512, 3, pad=1, flip_filters=False)
	net['conv5_2'] = ConvLayer(
		net['conv5_1'], 512, 3, pad=1, flip_filters=False)
	net['conv5_3'] = ConvLayer(
		net['conv5_2'], 512, 3, pad=1, flip_filters=False)
	net['pool5'] = PoolLayer(net['conv5_3'], 2)
	net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
	net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
	net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
	net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
	net['fc8'] = DenseLayer(
		net['fc7_dropout'], num_units=1000, nonlinearity=None)
	net['prob'] = NonlinearityLayer(net['fc8'], softmax)
	return net

net = build_model()
model = pkl.load(open('/home/nvidia/cnn_weights/lasagne/vgg16.pkl'))
output_layer = net['prob']
lasagne.layers.set_all_param_values(output_layer, model['param values'])
lasagne_params = lasagne.layers.get_all_params(net['prob'])
lasagne_names = [ par.name for par in lasagne_params ]

# for (lasagne_name, lasagne_param) in zip(lasagne_names, lasagne_params):
# 	lasagne_weight = lasagne_param.get_value()
# 	print lasagne_weight.shape, lasagne_weight.mean()

import torchvision.models as models
vgg16_torch_vision = models.vgg16(pretrained=True)
vgg16_torch_vision_dict = vgg16_torch_vision.state_dict()

# for param_name in vgg16_torch_vision_dict:
# 	print vgg16_torch_vision_dict[param_name].size(), vgg16_torch_vision_dict[param_name].mean()


# pdb.set_trace()

import torch
vgg_torch_caffe = torch.load('/home/nvidia/cnn_weights/pytorch/vgg16-00b39a1b.pth')
# for param in vgg_torch_caffe.keys():
	# print vgg_torch_caffe[param].size(), vgg_torch_caffe[param].mean()
print "number of lasagne params", len(lasagne_params)
print "number of pytorch pure params", len(vgg16_torch_vision_dict)
print "number of pytorch caffe params", len(vgg_torch_caffe.keys())

for (lasagne_param, pytorch_param, caffe_param) in zip(lasagne_params, vgg16_torch_vision_dict, vgg_torch_caffe.keys()):
	print "lasagne {:>20}  pytorch_pure {:>35} pytorch_caffe {:>35}  <===>   L : {:>8.4f}  PurePyTorch :  {:>8.4f} PytorchCaffe : {:>8.4f}".format(lasagne_param.get_value().shape, \
		vgg16_torch_vision_dict[pytorch_param].size(), vgg_torch_caffe[caffe_param].size(), \
		lasagne_param.get_value().mean(),  vgg16_torch_vision_dict[pytorch_param].mean(), vgg_torch_caffe[caffe_param].mean())


