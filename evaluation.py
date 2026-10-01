import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)

def evaluate_classifier(x_test, y_test, run_id):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = model.predict(x_test)
    
    acc = accuracy_score(y_test, preds)
    prec_0 = precision_score(y_test, preds, pos_label=0)
    rec_0 = recall_score(y_test, preds, pos_label=0)
    f1_0 = f1_score(y_test, preds, pos_label=0)
    prec_1 = precision_score(y_test, preds, pos_label=1)
    rec_1 = recall_score(y_test, preds, pos_label=1)
    f1_1 = f1_score(y_test, preds, pos_label=1)
    f1_macro = f1_score(y_test, preds, average='macro')
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_0", prec_0)
        mlflow.log_metric("recall_0", rec_0)
        mlflow.log_metric("f1_0", f1_0)
        mlflow.log_metric("precision_1", prec_1)
        mlflow.log_metric("recall_1", rec_1)
        mlflow.log_metric("f1_1", f1_1)
        mlflow.log_metric("f1_macro", f1_macro)
    
    print(f"Classifier Evaluation:")
    print(f"Recall-0    : {rec_0:.4f}")
    print(f"Precision-1 : {prec_1:.4f}")
    print(f"F1 Macro    : {f1_macro:.4f}")
    
    return f1_macro, rec_0, prec_1


def evaluate_regressor(x_test, y_test, run_id):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = model.predict(x_test)
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
    
    print(f"Regressor Evaluation:")
    print(f"R^2   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    
    return r2, rmse, mae

if __name__ == "__main__":
    evaluate_classifier()
    evaluate_regressor()