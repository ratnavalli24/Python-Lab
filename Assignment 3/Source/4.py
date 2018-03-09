from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
irisdataset=datasets.load_iris()
x=irisdataset.data
y=irisdataset.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model= KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
k_range=range(1,30)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt
plt.plot(k_range,scores)
plt.xlabel("K value")
plt.ylabel("Accuracy/Testing")
plt.show()