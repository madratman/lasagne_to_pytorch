### pretty printing is pretty ###
class Colors:
	def __init__(self):
		self.RED = '\033[91m'
		self.BLUE = '\033[94m'
		self.BOLD = '\033[1m'
		self.GREEN = '\033[92m'
		self.ENDC = '\033[0m'        
		self.LINE = "%s%s##############################################################################%s" \
								% (self.BLUE, self.BOLD, self.ENDC)
		self.PURPLE = '\033[95m'
		self.YELLOW = '\033[93m'
		self.UNDERLINE = '\033[4m'