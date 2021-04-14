import numpy as np 
import pickle 

class MatrixParser():
	def __init__(self, fname):
		f = open(fname)
		rows = f.readline().strip().split(',')[1:] # tester names 
		gids = []
		self.mtx = []
		for line in f:
			line = line.strip().split(',')
			# record gid 
			gid = line[0]
			gids.append(gid)
			# record values into matrix 
			values = line[1:]
			values = [float(i) for i in values]
			self.mtx.append(values)
		self.mtx = np.float64(self.mtx)

	def normalize(self):
		if hasattr(self, 'geo_mean') and hasattr(self, 'med'):
			result = self.mtx / self.geo_mean
			self.mtx_normed = result / self.med
		else:
			valid = (self.mtx>0).astype(np.float64).sum(axis=1, keepdims=True)
			log = np.log(self.mtx, where=self.mtx>0)
			log_sum = np.sum(log, axis=1, keepdims=True)
			log_sum = log_sum / valid
			self.geo_mean = np.exp(log_sum)
			
			result = self.mtx / self.geo_mean
			self.med = np.median(result, axis=0, keepdims=True)
			self.mtx_normed = result / self.med
		return self.mtx_normed

	def save(self, path):
		if hasattr(self, 'med') and hasattr(self, 'geo_mean'):
			pickle.dump({'med':self.med, 'geo_mean':self.geo_mean}, open(path, 'wb'))
		else:
			raise Exception('The normalization has not run yet')

	def restore(self, path):
		data = pickle.load(open(path, 'rb'))
		self.med = data['med']
		self.geo_mean = data['geo_mean']
		print('Successfully restored from:', path)
			
parser = MatrixParser('combined_251.csv')
mtx_norm = parser.normalize()
print(mtx_norm)
