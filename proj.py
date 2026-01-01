import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('/content/project database ml.csv')
    df.head()
    print(f"Dataset Shape: {df.shape}")
    print("File loaded successfully!")
except FileNotFoundError:
    print("ERROR: File not found at '/content/project database ml.csv'")
    print("Please check if the file name in the folder matches exactly.")
    raise

if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

df['avg_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
df['tenure_year'] = df['tenure'] // 12


le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print("Applying SMOTE to balance data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


print("Training Model...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train_smote, y_train_smote)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

train_pred = model.predict(X_train_smote)
train_acc = accuracy_score(y_train_smote, train_pred)

print("\n" + "="*40)
print("       FINAL PROJECT REPORT       ")
print("="*40)
print(f"Dataset Shape     : {df.shape}")
print(f"Training Accuracy : {train_acc*100:.2f}% ")
print(f"Test Accuracy     : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC Score     : {roc_auc_score(y_test, y_prob):.4f}")
print("-" * 40)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10)
plt.title('Top Factors Predicting Churn')
plt.show()

eval_set = [(X_train_smote, y_train_smote), (X_test, y_test)]
model.fit(X_train_smote, y_train_smote, eval_set=eval_set, verbose=False)


results = model.evals_result()
epochs = len(results['validation_0']['logloss']) 
x_axis = range(0, epochs)


plt.figure(figsize=(10, 6))

plt.plot(x_axis, results['validation_0']['logloss'], label='Training Loss')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test Loss')
plt.legend()
plt.ylabel('Log Loss (Lower is Better)')
plt.xlabel('Number of Trees')
plt.title('Learning Curve: Training vs Test Loss')
plt.show()

print("Interpreting the Graph:")
print("- If lines go down together: Good Fit ")
print("- If Training line goes down but Test line goes UP: Overfitting")
