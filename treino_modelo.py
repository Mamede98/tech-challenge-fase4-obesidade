import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv('Obesity.csv')
print("A iniciar o pré-processamento dos dados...")

colunas_para_arredondar = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
df[colunas_para_arredondar] = df[colunas_para_arredondar].round().astype(int)

X = df.drop('Obesity', axis=1)
y = df['Obesity']

label_encoders = {}
colunas_categoricas = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in colunas_categoricas:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

scaler = StandardScaler()
colunas_numericas = ['Age', 'Height', 'Weight']
X[colunas_numericas] = scaler.fit_transform(X[colunas_numericas])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("A treinar o modelo Random Forest...")
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

y_pred = modelo_rf.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f'\n[RESULTADO] Acurácia do Modelo: {acuracia * 100:.2f}%\n')

print('Relatório de Classificação detalhado:')
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

joblib.dump(modelo_rf, 'modelo_obesidade.pkl')
joblib.dump(scaler, 'scaler_numerico.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(le_target, 'target_encoder.pkl')

print("Artefatos guardados com sucesso! (.pkl)")