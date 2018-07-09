import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('voice.csv')
df_shuffled = df.sample(frac=1).reset_index(drop=True) #para embralhar a base

df_features = df_shuffled.drop('label', axis=1)
df_labels = df_shuffled['label'].values

features_treino, features_teste, labels_treino, labels_teste = train_test_split (df_features, df_labels, test_size=0.4, random_state=0)

# scaler = StandardScaler()
# scaler.fit(features_treino)
# StandardScaler(copy=True, with_mean=True, with_std=True)
# features_treino = scaler.transform(features_treino)
# features_teste = scaler.transform(features_teste)

clf = svm.SVC(kernel='rbf', C=1e3)
clf = clf.fit(features_treino, labels_treino)

resultados = clf.predict(features_teste)

i = 0
acertos = 0
erros = 0
for r in resultados:
    if r == labels_teste[i]:
        acertos = acertos + 1
    else:
        erros = erros + 1
    #print "Predict: ", r, " Expected: ", labels_teste[i]
    i = i + 1
print "|SVM - Final Score| - ", (acertos*100)/resultados.size, "%"
print "Acertos: ", acertos, "   Erros: ", erros, "  Total: ", resultados.size, "  Score:", clf.score(features_teste, labels_teste)

clf = tree.DecisionTreeClassifier() 
clf = clf.fit(features_treino, labels_treino) 

resultados = clf.predict(features_teste)

i = 0
acertos = 0
erros = 0
for r in resultados:
    if r == labels_teste[i]:
        acertos = acertos + 1
    else:
        erros = erros + 1
    #print "Predict: ", r, " Expected: ", labels_teste[i]
    i = i + 1
print "|TREE - Final Score| - ", (acertos*100)/resultados.size, "%"
print "Acertos: ", acertos, "   Erros: ", erros, "  Total: ", resultados.size, "  Score:", clf.score(features_teste, labels_teste)

scaler = StandardScaler()
scaler.fit(features_treino)
StandardScaler(copy=True, with_mean=True, with_std=True)
features_treino = scaler.transform(features_treino)
features_teste = scaler.transform(features_teste)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(features_treino, labels_treino)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
            hidden_layer_sizes=(30,30,30), learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9,
            nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
            verbose=False, warm_start=False)

resultados = mlp.predict(features_teste)

i = 0
acertos = 0
erros = 0
for r in resultados:
    if r == labels_teste[i]:
        acertos = acertos + 1
    else:
        erros = erros + 1
    #print "Predict: ", r, " Expected: ", labels_teste[i]
    i = i + 1
print "|REDE NEURAL - Final Score| - ", (acertos*100)/resultados.size, "%"
print classification_report(labels_teste, resultados)
print "Acertos: ", acertos, "   Erros: ", erros, "  Total: ", resultados.size#, "  Score:", clf.score(features_teste, labels_teste)
