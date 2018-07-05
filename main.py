import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('voice.csv')
df_shuffle = df.sample(frac=1).reset_index(drop=True)

# train, test  = train_test_split (df, test_size=0.4)
# clf = svm.SVC(kernel='linear', C=1).fit(train)

a = df_shuffle.drop('label', axis=1)
test = df_shuffle['label'].values


# clf = tree.DecisionTreeClassifier() 
# clf = clf.fit(a, test)

# arr = clf.predict(testee)
# m = 0
# f = 0
# for a in arr:
#     if a == 'male':
#         m = m + 1
#     else:
#         f = f + 1
#     print a

# print m
# print f


X_train, X_test, y_train, y_test = train_test_split (a, test, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='poly', C=1e3).fit(X_train, y_train)

predicao = clf.predict(X_test)
counter = 0
acertos = 0
erros = 0
for p in predicao:
    if p == y_test[counter]:
        acertos = acertos + 1
    else:
        erros = erros + 1

    # print "disse: ", p
    # print "certo: ", y_test[counter]
    counter = counter + 1

print clf.score(X_test, y_test)
print "TOTAL ACERTOS: ", acertos
print "TOTAL ERROS: ", erros


# features = [[140, 1], [130, 1],
#            [150, 0], [170, 0]]
# labels = [0, 0, 1, 1] # 0 e maca e 1 e laranja

# # o classificador encontra padroes nos dados de treinamento
# clf = tree.DecisionTreeClassifier() # instancia do classificador
# clf = clf.fit(features, labels) # fit encontra padroes nos dados

# # iremos utilizar para classificar uma nova fruta
# print(clf.predict([[150, 0]]))