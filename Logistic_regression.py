from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import plt
iris = datasets.load_iris()
feature = iris["data"][:,3:]
leabel =(iris["target"]==2).astype(np.int)

#training
clf = LogisticRegression()
clf.fit(feature,leabel)
example = clf.predict(([[1.2]]))
print(example)

#plot a graph
feature_new = np.linspace (0,3,1000).reshape(-1,1)
leabel_prob = clf.predict_proba(feature_new)
plt.plot(feature_new,leabel_prob[:,1],"g-",Label="verginica")
plt.show()



