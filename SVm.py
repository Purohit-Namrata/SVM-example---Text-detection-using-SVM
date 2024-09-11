import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
print(digits.data)   #actual data
print(digits.target)   #labels
clf = svm.SVC()

X,y = digits.data[:-10], digits.target[:-10]
clf.fit(X,y)
#print(clf.predict(digits.data[-5]))
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

print(digits.images[-5])