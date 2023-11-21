from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


class_ = pd.read_csv("class.csv")
zoo = pd.read_csv("zoo.csv")

print(zoo.class_type.value_counts())
plt.figure(figsize = (10,8))
sns.countplot(zoo.class_type)
plt.show()
data = zoo.copy()
data.drop("animal_name",axis = 1,inplace = True)
x = data.drop("class_type",axis = 1)# input data
y = data.class_type.values# target data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 42)
print("x_train shape : ",x_train.shape)
print("x_test shape : ",x_test.shape)
print("y_train shape : ",y_train.shape)
print("y_test shape : ",y_test.shape)

svm = SVC(random_state = 42,kernel = "linear")
svm.fit(x_train,y_train)
y_pred_svm = svm.predict(x_test)
print("Train Accuracy : ",svm.score(x_train,y_train))
print("Test Accuracy : ",svm.score(x_test,y_test))
cm_svm = confusion_matrix(y_test,y_pred_svm)
cr_svm = classification_report(y_test,y_pred_svm)
tp = cm_svm[0,0]
tn = cm_svm[1,1]
fp = cm_svm[0,1]
fn = cm_svm[1,0]

print("Sensitivity: ",tp/(tp + fn))
print("Specificity: ",tn/(tn + fp))

print("classification report : \n",cr_svm)
plt.figure(figsize = (10,8))
sns.heatmap(cm_svm,annot = True,cmap = "Blues",xticklabels = np.arange(1,8),yticklabels = np.arange(1,8))
plt.show()
