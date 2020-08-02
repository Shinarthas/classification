import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

# разодьем данные на тестовые и проверочные
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)#random_state=1 убирает рандом что он всегжа одинаков


# feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() #сколько среднеквадратичных отклонений содержит наша величина
X_train=ss.fit_transform(X_train)#применяем к тестовой выборке
# когда мы вызываем fit_transform мы (1) готовим модель кторая конвертирует, а потом на основе ее изменяем наши данные
X_test=ss.transform(X_test) # тут только transform потому что мы ТОЛЬКО ЧТО создали модель странсформации, и среднее и отклонение УЖЕ расчитаны, поэтому


#SVM classification on the testing set
from sklearn.svm import SVC
lr=SVC(kernel='linear',random_state=0)
lr.fit(X_train,y_train)

result=lr.predict(ss.transform([[60,65000]]))
print(result)


y_pred=lr.predict(X_test)

np.set_printoptions()
print(np.concatenate(
    (y_test.reshape(len(y_test), 1),
     y_pred.reshape(len(y_pred), 1)
     ),
    1))


# making confusion matrix
# количество правильных и не правильных предсказаний
from sklearn.metrics import confusion_matrix, accuracy_score
# вернет матрицу 2x2 где будет количество верно угаданных позитивных ответов и не угаданных позитивных
# [[верно предсказанные положительные] [не верно предсказанные положительные]
# [не верно предсказанные отрицательные] [верно предсказанные отрицательные]]
# accuracy_score -- сколько верных предсказания
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred)) # вернет от 0 до 1



