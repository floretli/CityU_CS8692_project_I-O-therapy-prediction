from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
import pickle
import pandas as pd
from sklearn.decomposition import NMF

def train(X,y):   ##X=datalist, y =labellist
	svr = GridSearchCV(SVR(kernel='rbf', gamma=0.001),
					   param_grid={ "C": [1e-3,1e-2,1e-1,1e0, 1e1, 1e2, 1e3],
								   "epsilon":[1e-3,1e-2,1e-1,1e0, 1e1, 1e2, 1e3]
								   })

	#svr = SVR()

	svr.fit(X, y)
	means = svr.cv_results_['mean_test_score']
	params = svr.cv_results_['params']
	for mean, param in zip(means, params):
		print("%f  with:   %r" % (mean, param))
	return svr


class SampleLabel():
	def __init__(self, label_file):
		f=open(label_file,'r')
		self.smp_pfs={}
		self.smp_os={}
		self.smp_bor={}
		self.smp_group={}
		for line in (f.readlines())[1:]: 
			line.strip()
			line_ele = line.split(",")
			(name,pfs,os,bor,group) = (line_ele[0],float(line_ele[8]),float(line_ele[1]),int(line_ele[7]),int(line_ele[10]))
			# if( os==-1):
			# 	pass
			#
			# if (bor=='PD'):
			# 	bor=1;
			# elif(bor=='NON-PD'):
			# 	bor=0;
			# else:
			# 	print ("error! check bor value of smp:",name)

			self.smp_pfs[name]=pfs
			if (os!=-1):
				self.smp_os[name]=os 
			self.smp_bor[name]=bor
			self.smp_group[name]=group
	def getpfs(self):
		return self.smp_pfs
	def getos(self):
		return self.smp_os
	def getbor(self):
		return self.smp_bor
	def getgroup(self):
		return self.smp_group

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



if __name__=='__main__':

	gene_csv = "Raiz_fpkm_geneid_rmdup51smp_19063.csv"
	label_csv = "Riaz2017_Nivo_Melanoma_clinical_group.csv"
	#label_csv = "./HyePrat_41smp_clinical.csv"
	model_file = "./delta2-antipd1-pfs-model.bin"

	print ("gene_csv,label_csv ", gene_csv,label_csv)

	parser = MatrixParser(gene_csv)



	#gene_mtx_norm = parser.normalize()
	smp_exprs = parser.getlist()   ## a dict: smp_exprs[smp] = [gene value list]
	smp_names =smp_exprs.keys()
	

	##get labels
	labels= SampleLabel(label_csv)
	pfs_dict=labels.getpfs()   ## adict: pfs_dict[smp] = float(pfsvalue)
	group_dict=labels.getgroup()

	## select group
	X_train,y_train,X_check,y_check= ([],[],[],[])


	slect_group=random.randint(1,9)
	print("test group=",slect_group)
	for name in smp_names:
		if (group_dict[name]!=slect_group):
			X_train.append(smp_exprs[name])
			y_train.append(pfs_dict[name])
		else:
			X_check.append(smp_exprs[name])
			y_check.append(pfs_dict[name])	


	check=pd.DataFrame(X_check)
	check.transpose().to_csv("./normal.csv")

	model_nmf = NMF(n_components=10, init='random', random_state=0)

	X_train = model_nmf.fit_transform(X_train)
	X_check = model_nmf.fit_transform(X_check)


	pfs_svr=train(X_train,y_train)

	#test svr model
	y_predict = pfs_svr.predict(X_check)

	print ("predict results:",y_predict)
	print ("true results:",y_check)

	##save the model
	
	savemodel=open(model_file,"wb")
	pickle.dump(pfs_svr,savemodel)
