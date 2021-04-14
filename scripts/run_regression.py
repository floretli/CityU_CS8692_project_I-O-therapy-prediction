from sklearn.svm import SVR
import numpy as np
import random
import sys, os
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import RadiusNeighborsRegressor as RR
from sklearn.neural_network import MLPRegressor as MLP 

class MatrixParser():
	def __init__(self, fname):
		f = open(fname)
		self.rows = f.readline().strip().split(',')[1:] # tester names 
		self.gids = []  ## gene ids
		self.mtx = []
		for line in f:
			line = line.strip().split(',')
			# record gid 
			gid = line[0]
			self.gids.append(gid)
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
			self.mtx_normed =  self.mtx / self.med
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

	def getlist(self):
		smp_exprs = {}  ## each sample has a gene list
		gene_mtx_norm = self.normalize()
		check=pd.DataFrame(gene_mtx_norm)
		check.transpose().to_csv("./normal2.csv")
		for i in range(len(self.rows)):
			smp_exprs[self.rows[i]]=[ gene_mtx_norm[k][i] for k in range(len(self.gids))]
		return smp_exprs

def random_validation_set(patient_list, ratio=0.1):
	valid_list = random.sample(patient_list, len(patient_list)//10)
	return valid_list

def split_train_val(data, name_list, valid_list):
	idx_train = []
	name_train = []
	idx_val = []
	name_val = []
	for i,name in enumerate(name_list):
		if name in valid_list:
			idx_val.append(i)
			name_val.append(name)
		else:
			idx_train.append(i)
			name_train.append(name)
	train_mtx = data[:,np.int64(idx_train)]
	val_mtx = data[:, np.int64(idx_val)]
	return train_mtx, val_mtx, name_train, name_val

def get_clinical_data(fname):
	f = open(fname)
	f.readline()
	label_dict = {}
	for line in f:
		line = line.strip().split(',')
		name, pfs = line[0], float(line[14])
		label_dict[name] = pfs
	return label_dict

def get_train_val_label(label_dict, name_train, name_val):
	pfs_train = []
	pfs_val = []
	for name in name_train:
		pfs = label_dict[name]
		pfs_train.append(pfs)
	for name in name_val:
		pfs = label_dict[name]
		pfs_val.append(pfs)
	pfs_train = np.float32(pfs_train)
	pfs_val = np.float32(pfs_val)
	return pfs_train, pfs_val

# gene_csv = './combined_251.csv'
# label_csv = 'HwangHyePrat_clinical.csv'
gene_csv = 'Hye2020_Prat2017_count_755.csv'
label_csv = 'HyePrat_41smp_clinical.csv'

parser = MatrixParser(gene_csv)
valid_name_list = random_validation_set(parser.rows) # random select the validation name list 
train_mtx, val_mtx, name_train, name_val = split_train_val(parser.normalize(), parser.rows, valid_name_list)
train_mtx = train_mtx.T 
val_mtx = val_mtx.T

label_dict = get_clinical_data(label_csv)
pfs_train, pfs_val = get_train_val_label(label_dict, name_train, name_val)


# pca = PCA(n_components=5)
# train_mtx = pca.fit_transform(train_mtx)
# val_mtx = pca.transform(val_mtx)
train_mtx = train_mtx / 100
val_mtx = val_mtx / 100

svr = SVR(kernel='rbf', gamma=0.0001, tol=1e-5)
svr.fit(train_mtx, pfs_train)
res_train = svr.predict(train_mtx)
res_val = svr.predict(val_mtx)

# rr = RR()
# rr.fit(train_mtx, pfs_train)
# res_train = rr.predict(train_mtx)
# res_val = rr.predict(val_mtx)

# mlp = MLP(hidden_layer_sizes=(500,100), max_iter=200, activation='tanh', alpha=0.0001)
# mlp.fit(train_mtx, pfs_train)
# res_train = mlp.predict(train_mtx)
# res_val = mlp.predict(val_mtx)

print('PRED:', res_train)
print('TRAIN:' , pfs_train)

print('PRED:', res_val)
print('VAL:' , pfs_val)
