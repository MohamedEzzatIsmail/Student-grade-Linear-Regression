import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

file = pd.read_csv('student-mat.csv', sep=';')
file = file[["G1", "G2", "G3", "studytime", "failures", "absences"]]
perdict = "G3"
X = np.array(file.drop(columns=[perdict]))
Y = np.array(file[perdict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

b = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    Linear = linear_model.LinearRegression()
    Linear.fit(x_train, y_train)
    acc = Linear.score(x_test, y_test)
    print("acc = " + str(acc))
    if acc > b :
        b = acc
        with open("model.pickle", "wb") as f:
            pickle.dump(Linear, f)



p = open("model.pickle", "rb")
Linear = pickle.load(p)

predict = Linear.predict(x_test)

for x in range(len(predict)):
    print(predict[x], x_test[x], y_test[x])

style.use('ggplot')
plt.scatter(file['absences'], file['G3'])
plt.xlabel("G1")
plt.ylabel("G3")
plt.show()
