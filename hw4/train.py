from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import csv
import sys

print(sys.argv[1],sys.argv[2])

data = []
check_index = []

with open(sys.argv[1]+'title_StackOverflow.txt') as f:
    data = f.readlines()
data = np.array(data)

with open(sys.argv[1]+'check_index.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        check_index.append(row)
check_index = np.array(check_index)
check_index = np.delete(check_index,0,0)

print(check_index.shape)

vectorizer = TfidfVectorizer(max_df=0.5, max_features=None, min_df=2, stop_words='english')
X = vectorizer.fit_transform(data)



svd = TruncatedSVD(n_components=20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

#K-means
km = KMeans(n_clusters=85, init='k-means++', n_init=20, verbose=False).fit(X)
#print('end')
print(X.shape)
L = km.labels_
print(L.shape)

count = 0

with open(sys.argv[2],'w') as f:
    w = csv.writer(f)
    w.writerows([['ID','Ans']])
    for i in range(check_index.shape[0]):
    	a = int(check_index[i][1])
    	b = int(check_index[i][2])
    	if L[a]== L[b]:
        	w.writerows([[i,1]])
        	count += 1
        else:
        	w.writerows([[i,0]])
print('alike:',count)