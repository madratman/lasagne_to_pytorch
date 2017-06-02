import lasagne
import cPickle as pkl
import pdb
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax
import io
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
from lasagne.utils import floatX

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
pretrained_model = pkl.load(open('/home/nvidia/cnn_weights/lasagne/vgg16.pkl'))
# pretrained_model_2 = pkl.load(open('/home/nvidia/cnn_weights/lasagne/vgg_cnn_s.pkl'))

output_layer = net['prob']
lasagne.layers.set_all_param_values(output_layer, pretrained_model['param values'])
lasagne_params = lasagne.layers.get_all_params(net['prob'])
lasagne_names = [ par.name for par in lasagne_params ]

MEAN_IMAGE = pretrained_model['mean value'] # this is  [103.939, 116.779, 123.68]. 
MEAN_IMAGE = MEAN_IMAGE.reshape([3,1,1])
# MEAN_IMAGE = pretrained_model_2['mean image'] # this is (3,224,224)

def prep_image(url):
	ext = url.split('.')[-1]
	im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
	# Resize so smallest dim = 256, preserving aspect ratio
	h, w, _ = im.shape
	if h < w:
		im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
	else:
		im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
	# Central crop to 224x224
	h, w, _ = im.shape
	im = im[h//2-112:h//2+112, w//2-112:w//2+112]
	rawim = np.copy(im).astype('uint8')
	# Shuffle axes to c01
	im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
	# Convert to BGR
	im = im[::-1, :, :]
	im = im - MEAN_IMAGE
	return rawim, floatX(im[np.newaxis])

import urllib

index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
image_urls = index.split('<br>')

rawim, im = prep_image(image_urls[0])

prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
prob = np.array(lasagne.layers.get_output(net['conv1_1'], im, deterministic=True).eval())
top5 = np.argsort(prob[0])[-1:-6:-1]
