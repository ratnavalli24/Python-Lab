from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
A=datasets.load_iris()
i=A.data
l=A.target
atrain, atest, btrain, btest=train_test_split(i,l,test_size=0.2,random_state=20)
atrain1,atest1,btrain1,btest1=train_test_split(i,l,test_size=0.2,random_state=20)
s=SVC(kernel='linear')
s1=SVC(kernel='rbf')
s.fit(atrain, btrain)
bpred=s.predict(atest)
print("Accuracy Score/Linear Kernal : ", accuracy_score(bpred, btest))
s1.fit(atrain1,btrain1)
bpred1=s.predict(atest1)
print("Accuracy Score/RBF Kernal : ",accuracy_score(bpred,btest1))