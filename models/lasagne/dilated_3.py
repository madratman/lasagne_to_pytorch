from lasagne.layers import DilatedConv2DLayer, PadLayer, InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from collections import OrderedDict
import lasagne
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer => cuda v/s libgpuarray backend issues
#https://github.com/Lasagne/Lasagne/issues/698#issuecomment-225924888
#https://github.com/Lasagne/Lasagne/pull/836

def build_model(height, width):
	net = OrderedDict()
	net['input'] = InputLayer((None, 3, height, width), name='input')
	net['conv1'] = ConvLayer(net['input'], num_filters=32, filter_size=7, pad='same', name='conv1')
	net['conv2'] = ConvLayer(net['conv1'], num_filters=32, filter_size=5, pad='same', name='conv2')
	net['conv3'] = ConvLayer(net['conv2'], num_filters=64, filter_size=3, pad='same', name='conv3')
	net['conv4'] = ConvLayer(net['conv3'], num_filters=64, filter_size=3, pad='same', name='conv4')

	net['pad5'] = PadLayer(net['conv4'], width=1, val=0, name='pad5')
	net['conv_dil5'] = DilatedConv2DLayer(net['pad5'], num_filters=64, filter_size=3, dilation=(1,1), name='conv_dil5')

	net['pad6'] = PadLayer(net['conv_dil5'], width=2, val=0, name='pad6')
	net['conv_dil6'] = DilatedConv2DLayer(net['pad6'], num_filters=64, filter_size=3, dilation=(2,2), name='conv_dil6')

	net['pad7'] = PadLayer(net['conv_dil6'], width=4, val=0, name='pad6')
	net['conv_dil7'] = DilatedConv2DLayer(net['pad7'], num_filters=64, filter_size=3, dilation=(4,4), name='conv_dil7')

	net['pad8'] = PadLayer(net['conv_dil7'], width=8, val=0, name='pad8')
	net['conv_dil8'] = DilatedConv2DLayer(net['pad8'], num_filters=64, filter_size=3, dilation=(8,8), name='conv_dil8')

	net['pad9'] = PadLayer(net['conv_dil8'], width=16, val=0, name='pad9')
	net['conv_dil9'] = DilatedConv2DLayer(net['pad9'], num_filters=64, filter_size=3, dilation=(16,16), name='conv_dil9')

	net['pad10'] = PadLayer(net['conv_dil9'], width=1, val=0, name='pad10')
	net['l_out'] = DilatedConv2DLayer(net['pad10'], num_filters=2, filter_size=3, dilation=(1,1), name='l_out')

	for layer in lasagne.layers.get_all_layers(net['l_out']):
		print layer.name,layer.output_shape
	print "output shape", net['l_out'].output_shape

	net['l_in'] = net['input']
	return net