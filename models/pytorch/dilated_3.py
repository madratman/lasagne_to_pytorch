import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import colors as col

class ConvNet(torch.nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.FIRST_CALL = True

		self.conv_1 = torch.nn.Conv2d(3, 32, kernel_size=7, padding=3)
		self.relu_1 = torch.nn.ReLU(inplace=True)

		self.conv_2 = torch.nn.Conv2d(32, 32, kernel_size=5, padding=2)
		self.relu_2 = torch.nn.ReLU(inplace=True)
		
		self.conv_3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.relu_3 = torch.nn.ReLU(inplace=True)

		self.conv_4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.relu_4 = torch.nn.ReLU(inplace=True)

		self.conv_5 = torch.nn.Conv2d(64, 64, padding=1, dilation=1, kernel_size=3)
		self.relu_5 = torch.nn.ReLU(inplace=True)

		self.conv_6 = torch.nn.Conv2d(64, 64, padding=2, dilation=2, kernel_size=3)
		self.relu_6 = torch.nn.ReLU(inplace=True)

		self.conv_7 = torch.nn.Conv2d(64, 64, padding=4, dilation=4, kernel_size=3)
		self.relu_7 = torch.nn.ReLU(inplace=True)

		self.conv_8 = torch.nn.Conv2d(64, 64, padding=8, dilation=8, kernel_size=3)
		self.relu_8 = torch.nn.ReLU(inplace=True)

		self.conv_9 = torch.nn.Conv2d(64, 64, padding=16, dilation=16, kernel_size=3)
		self.relu_9 = torch.nn.ReLU(inplace=True)

		self.conv_10 = torch.nn.Conv2d(64, 2, padding=1, dilation=1, kernel_size=3)
		self.relu_10 = torch.nn.ReLU(inplace=True)

		self.forward_pass = nn.Sequential(*[self.conv_1, self.relu_1, self.conv_2, self.relu_2, self.conv_3, self.relu_3, \
											self.conv_4, self.relu_4, self.conv_5, self.relu_5, self.conv_6, self.relu_6, \
											self.conv_7, self.relu_7, self.conv_8, self.relu_8, self.conv_9, self.relu_9, \
											self.conv_10, self.relu_10])

	def forward(self, x):
		# colors = col.Colors()
		# conv_1 = self.relu_1(self.conv_1(x))
		# conv_2 = self.relu_2(self.conv_2(conv_1))
		# conv_3 = self.relu_3(self.conv_3(conv_2))
		# conv_4 = self.relu_4(self.conv_4(conv_3))
		# conv_5 = self.relu_5(self.conv_5(conv_4))
		# conv_6 = self.relu_6(self.conv_6(conv_5))
		# conv_7 = self.relu_7(self.conv_7(conv_6))
		# conv_8 = self.relu_8(self.conv_8(conv_7))
		# conv_9 = self.relu_9(self.conv_9(conv_8))
		# conv_10 = self.relu_10(self.conv_10(conv_9))

		# if self.FIRST_CALL:
		# 	print "input.shape", colors.RED, x.size(), colors.ENDC 
		# 	print "conv_1.shape", colors.RED, conv_1.size(), colors.ENDC
		# 	print "conv_2.shape", colors.RED, conv_2.size(), colors.ENDC
		# 	print "conv_3.shape", colors.RED, conv_3.size(), colors.ENDC
		# 	print "conv_4.shape", colors.RED, conv_4.size(), colors.ENDC
		# 	print "conv_5.shape", colors.RED, conv_5.size(), colors.ENDC
		# 	print "conv_6.shape", colors.RED, conv_6.size(), colors.ENDC
		# 	print "conv_7.shape", colors.RED, conv_7.size(), colors.ENDC
		# 	print "conv_8.shape", colors.RED, conv_8.size(), colors.ENDC
		# 	print "conv_9.shape", colors.RED, conv_9.size(), colors.ENDC
		# 	print "conv_10.shape", colors.RED, conv_10.size(), colors.ENDC
		# 	self.FIRST_CALL = False

		return self.forward_pass(x)

def build_model():
	return ConvNet()
