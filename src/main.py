import preprocessing
import training
import evaluating
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

def main():
    #1: Preprocessing
    listings = pd.read_csv("data/listings.csv")
    
    train_df, test_df = preprocessing.preprocess_data(listings)
    
    feature_groups = preprocessing.get_feature_groups(train_df)
    
    #2: Training
    X_train, y_train = training.prepare_training_data(train_df, cap_low=500, cap_high=15000, feature_groups=feature_groups, use_log=True)
    X_test, y_test = training.prepare_training_data(test_df, cap_low=500, cap_high=15000, feature_groups=feature_groups, use_log=True)
    
    model = XGBRegressor(subsample=0.7, reg_lambda=5, reg_alpha=0.5, n_estimators=200, min_child_weight=3, max_depth=7, learning_rate=0.05, colsample_bytree=0.7, random_state=8)
    
    y_true, y_pred = training.evaluate_model(model, X_train, y_train, X_test, y_test, use_log=True)
    
    #3: Evalutation
    metrics = evaluating.evaluate_metrics(y_true, y_pred)
    print("Overall Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    
    bins = [0, 1000, 2000, 4000, 6000, 10000, np.inf]
    labels = ['<1000', '1000-2000', '2000-4000', '4000-6000', '6000-10000', '>10000']
    
    evaluating.evaluate_buckets(y_true, y_pred, bins, labels)
    
    evaluating.plot_results(y_true, y_pred, label='final')

if __name__ == "__main__":
    main()