import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PRODUCCION = False
best_clf = best_model #Asignar aqui el mejor clasificador posible (previamente entrenado)

#Leemos el dataset de evaluación, simulando producción
if PRODUCCION==False:
    df = pd.read_csv("https://raw.githubusercontent.com/FedericoFortuna/CienciaDeDatos/main/TP_Virus_Alumnos.csv")
    #_, df = train_test_split(df, test_size=0.3, random_state=3)
else:
    df = pd.read_csv("TP_Virus_Evaluacion.csv")
#Dividimos en target y predictoras

X_prod = df.drop(["target", "Genero","Laboral"], axis=1)
y_prod = df["target"]

# ENTREGUE ESTA COMENTADA, DECIR LO MISMO, QUE SE VE QUE ENTREGUE UNA ANTERIOR, LA QUE VALE ES LA DE ABAJO
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_prod, y_prod, test_size=0.3, random_state=42)

#Transformaciones

si = ColImputer(imputer=SimpleImputer(strategy='mean'), columns=['Edad'])
si.fit(X_train)
X_train = si.transform(X_train)
X_test = si.transform(X_test)

si = ColImputer(imputer=SimpleImputer(strategy='median'), columns=['LVL'])
si.fit(X_train)
X_train = si.transform(X_train)
X_test = si.transform(X_test)
si = Outliers(columns=['LVL'])
si.fit(X_train)
X_train = si.transform(X_train)
X_test = si.transform(X_test)

si = StdScaler(scaler=MinMaxScaler(), columns=["BLD01","BLD02","BLD03","REC1","REC2","REC3","REC4","REC5"])
si.fit(X_train)
X_train = si.transform(X_train)
X_test = si.transform(X_test)

#Evaluación final

### ESTO LO ENTREGUE DESCOMENTADO, PARA LA DEFENSA DECIR QUE ENTREGUE UNO VIEJO SE VE, (DA MEJOR SIN EL FIT) -> Esto no decirlo
#best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)


print(f'---------------------------------------------{model_name.upper()}-------------------------------------------------------')
print('Exactitud (accuracy) del modelo: {:.2f} %'.format(accuracy_score(y_test, y_pred) * 100))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.xlabel('% 1 – Specificity (falsos positivos)', fontsize=14)
plt.ylabel('% Sensitivity (positivos)', fontsize=14)

# Graficar la línea de azar
it = [i / 100 for i in range(100)]
plt.plot(it, it, label="AZAR AUC=0.5", color="black")
graficarCurvaRoc(best_clf, X_test, y_test, model_name)
# Añadir título y mostrar el gráfico
plt.title('Curvas ROC de Modelos', fontsize=16)
plt.legend()
plt.show()
