import mlflow
import dagshub
from sklearn.linear_model import LogisticRegression

# ربط الكود بـ DagsHub
dagshub.init(repo_owner='adhamelnaggar354', repo_name='my-first-repo', mlflow=True)

# بدء الـ Parent run
with mlflow.start_run(run_name="Hyperparameter_Tuning"):
    
    # أول تجربة (Nested Run)
    with mlflow.start_run(run_name="C_0.1", nested=True):
        mlflow.log_param("C", 0.1)
        mlflow.log_metric("accuracy", 0.85)
        
    # ثاني تجربة (Nested Run)
    with mlflow.start_run(run_name="C_1.0", nested=True):
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", 0.89)

print("✅ Tuning experiments logged to DagsHub!")