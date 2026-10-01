import pandas as pd
from data_ingestion import ingest_data
from train import train_classifier, train_regressor
from evaluation import evaluate_classifier, evaluate_regressor
from sklearn.model_selection import train_test_split

# Classifier thresholds
F1_MACRO_THRESHOLD = 0.65
RECALL_0_THRESHOLD = 0.80
PRECISION_1_THRESHOLD = 0.90

# Regressor thresholds
R2_THRESHOLD = 0.75
RMSE_THRESHOLD = 1.5
MAE_THRESHOLD = 1.5


def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    df = pd.read_csv("ingested/A_join.csv")

    df['placement_status'] = df['placement_status'].replace({
        'Not Placed': 0,
        'Placed': 1
    })
    
    X = df.drop(columns=['Student_ID', 'placement_status', 'salary_lpa'])
    y_placement = df['placement_status']
    y_salary = df['salary_lpa']
    
    X_train, X_test, y_place_train, y_place_test, y_sal_train, y_sal_test = train_test_split(
        X, y_placement, y_salary,
        test_size=0.2, random_state=42, stratify=y_placement
    )
    
    print("Step 2: Training Classifier")
    clf_run_id = train_classifier(X_train, y_place_train)
    
    print("Step 3: Evaluating Classifier")
    f1_macro, rec_0, prec_1  = evaluate_classifier(X_test, y_place_test, clf_run_id)
    
    # Regresi
    placed_train = y_place_train == 1
    placed_test = y_place_test == 1
    
    print("Step 4: Training Regressor")
    reg_run_id = train_regressor(X_train[placed_train], y_sal_train[placed_train])
    
    print("Step 5: Evaluating Regressor")
    r2, rmse, mae = evaluate_regressor(
        X_test[placed_test], y_sal_test[placed_test], reg_run_id
    )

    clf_approved = (
        f1_macro >= F1_MACRO_THRESHOLD 
        and rec_0 >= RECALL_0_THRESHOLD 
        and prec_1 >= PRECISION_1_THRESHOLD
    )
    
    reg_approved = (
        r2 >= R2_THRESHOLD 
        and rmse <= RMSE_THRESHOLD 
        and mae <= MAE_THRESHOLD
    )
    
    if clf_approved and reg_approved:
        print("Both models approved for deployment.")
    elif clf_approved and not reg_approved:
        print("Only classifier approved. Regressor needs review.")
    elif not clf_approved and reg_approved:
        print("Only regressor approved. Classifier needs review.")
    else:
        print("Both models rejected. Review before deployment.")
    

if __name__ == "__main__":
    run_pipeline()