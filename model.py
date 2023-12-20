import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('iris.csv')
print(df.head(5))
#select independent and dependent variable
X= df[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']]
y = df['Class']

#split the dataset into train and test
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=50)

#Feature Scaling
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Instantiate the model
classifier= RandomForestClassifier()

#fit the model

classifier.fit(X_train,y_train)





#make the pickle file of the model
pickle.dump(classifier, open("model.pkl","wb"))

