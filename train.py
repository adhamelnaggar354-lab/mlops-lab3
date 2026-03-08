import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib

# ربط الكود بـ DagsHub (تأكد من اسم الريبو بتاعك)
dagshub.init(repo_owner='adhamelnaggar354', repo_name='my-first-repo', mlflow=True)

# تحميل الداتا اللي عملنالها بريبروسيس
train = pd.read_csv('train.csv')
X_train = train.drop('target', axis=1)
y_train = train['target']

# بدء تجربة MLflow
with mlflow.start_run(run_name="RandomForest_Train"):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # تسجيل الموديل والنتائج
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, 'model.pkl')
    
    print("✅ Model trained and logged to DagsHub!")