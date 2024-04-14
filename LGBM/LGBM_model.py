import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold

def train_lightgbm_small(df, cat_features, sample_frac=0.02, test_size=0.2, num_boost_round=1000, early_stopping_rounds=100, threshold=0.5):
    X = df.drop(['target'], axis=1) 
    y = df['target']

    # Encode categorical variables
    # label_encoders = {}
    # for cat_col in cat_features:
    #     label_encoders[cat_col] = LabelEncoder()
    #     X[cat_col] = label_encoders[cat_col].fit_transform(X[cat_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create the LightGBM data containers
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, free_raw_data=False, reference=train_data)

    # Sampling a fraction of the data
    sampled_indices = y_train.sample(frac=sample_frac, random_state=42).index
    X_train_sample = X_train.loc[sampled_indices]
    y_train_sample = y_train.loc[sampled_indices]

    train_data_sample = lgb.Dataset(X_train_sample, label=y_train_sample, categorical_feature=cat_features, free_raw_data=False)

    def lgbm_eval(num_leaves, learning_rate, feature_fraction, bagging_fraction):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": int(num_leaves),
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "verbose": -1
            #"device":'gpu'
        }

        cv_result = lgb.cv(params, train_data_sample, num_boost_round=num_boost_round, nfold=5, 
                           stratified=False, seed=42, metrics='auc')

        return max(cv_result['valid auc-mean'])

    # Define the parameter bounds for Bayesian Optimization
    param_bounds = {
        'num_leaves': (20, 40),
        'learning_rate': (0.01, 0.2),
        'feature_fraction': (0.8, 1.0),
        'bagging_fraction': (0.8, 1.0),
    }

    # Bayesian Optimization
    optimizer = BayesianOptimization(f=lgbm_eval, pbounds=param_bounds, random_state=42)

    # Optimization
    optimizer.maximize(init_points=5, n_iter=25)

    print("Best AUC: ", optimizer.max['target'])
    print("Best parameters: ", optimizer.max['params'])

    params = optimizer.max['params']
    params['num_leaves'] = int(params['num_leaves'])
    params['objective'] = 'binary'
    params['boosting'] = 'gbdt'
    params['metric'] = 'auc'

    # Use GPU
    #params['device'] = 'gpu'

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=True)]
    )

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    auc = roc_auc_score(y_test, y_pred)
    print(f'The ROC AUC on test set is: {auc:.4f}')

    # Convert to binary
    y_pred_binary = (y_pred > threshold).astype(int)

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_pred_binary)
    print(f'The accuracy on the test set is: {test_accuracy:.4f}')

    # Predict on the training set
    y_train_pred = model.predict(X_train)

    # Convert training set predictions to binary outcomes
    y_train_pred_binary = (y_train_pred > threshold).astype(int)

    # Calculate accuracy on the training set
    train_accuracy = accuracy_score(y_train, y_train_pred_binary)
    print(f'The accuracy on the training set is: {train_accuracy:.4f}')

    return model

def train_lightgbm(df, cat_features, sample_frac=0.02, test_size=0.2, num_boost_round=1000, early_stopping_rounds=100, threshold=0.5,num_folds=5,):
    X = df.drop(['target'], axis=1) 
    y = df['target']

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    def lgbm_eval(num_leaves, learning_rate, feature_fraction, bagging_fraction):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": int(num_leaves),
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "verbose": -1
        }

        # Store AUC scores for each fold
        fold_aucs = []

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
            valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features, free_raw_data=False, reference=train_data)

            # Train with early stopping
            model = lgb.train(params, train_data, num_boost_round=num_boost_round,
                              valid_sets=[valid_data])

            # Evaluate on the validation set
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            fold_auc = roc_auc_score(y_valid, y_pred)
            fold_aucs.append(fold_auc)

        return np.mean(fold_aucs)

    # Bayesian Optimization
    param_bounds = {
        'num_leaves': (20, 40),
        'learning_rate': (0.01, 0.2),
        'feature_fraction': (0.8, 1.0),
        'bagging_fraction': (0.8, 1.0),
    }

    optimizer = BayesianOptimization(f=lgbm_eval, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=25)

    # Print best AUC and parameters found
    print("Best AUC: ", optimizer.max['target'])
    print("Best parameters: ", optimizer.max['params'])

    # Train final model on the full dataset with best parameters
    best_params = optimizer.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['boosting_type'] = 'gbdt'

    train_data = lgb.Dataset(X, label=y, categorical_feature=cat_features, free_raw_data=False)
    final_model = lgb.train(best_params, train_data, num_boost_round=num_boost_round)

    return final_model

def train_lightgbm_kfold(df, cat_features, num_folds=5, num_boost_round=1000, early_stopping_rounds=100):
    X = df.drop(['target'], axis=1)
    y = df['target']

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Parameters for LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }

    # List to store fold-wise performance
    fold_aucs = []

    # Iterate over each fold
    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # Create LightGBM data containers
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
        valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features, free_raw_data=False, reference=train_data)

        # Train the model
        model = lgb.train(params, train_data, num_boost_round=num_boost_round,
                          valid_sets=[valid_data], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=True)])

        # Predict on the validation set
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

        # Compute the ROC AUC Score
        fold_auc = roc_auc_score(y_valid, y_pred)
        fold_aucs.append(fold_auc)
        print(f"Fold {len(fold_aucs)} AUC: {fold_auc}")

    # Calculate average AUC over all folds
    mean_auc = np.mean(fold_aucs)
    print(f"Mean ROC AUC: {mean_auc}")

    # Optional: Retrain model on the full dataset
    full_train_data = lgb.Dataset(X, label=y, categorical_feature=cat_features, free_raw_data=False)
    final_model = lgb.train(params, full_train_data, num_boost_round=num_boost_round)

    return final_model


if __name__ == "__main__":
    #df_1 = pd.read_csv('data/train_data.csv')
    #df_label = pd.read_csv('data/train_labels.csv')
    #df = pd.merge(df_1, df_label, on='customer_ID', how='inner')


    #df = pd.read_parquet('data/train.parquet')

    df = pd.read_csv('LGBM_data/train_data_encoded.csv')
    #df = df.drop(['customer_ID', 'S_2'], axis=1)
    df = df.drop(['customer_ID'], axis=1)

    # Find the columns that have missing value
    missing_value_columns = df.isnull().sum()

    # Check if there are more missing values
    #missing_value_columns = missing_value_columns[missing_value_columns > 0]
    #print(missing_value_columns)

    cat_features = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126",
    "D_63", "D_64", "D_66", "D_68"
    ]

    # # Impute categorical features with the most frequent value
    # cat_imputer = SimpleImputer(strategy='most_frequent') 
    # df[cat_features] = cat_imputer.fit_transform(df[cat_features])

    # # Get list of numerical features
    # num_features = [col for col in df.columns if col not in cat_features + ['customer_ID']]

    # # Impute numerical features with the mean
    # num_imputer = SimpleImputer(strategy='mean')
    # df[num_features] = num_imputer.fit_transform(df[num_features])

    # # Find the columns that have missing value
    # missing_value_columns = df.isnull().sum()

    # # Check if there are more missing values
    # missing_value_columns = missing_value_columns[missing_value_columns > 0]
    # print(missing_value_columns)

    model = train_lightgbm_small(df, cat_features)

    test_df = pd.read_csv('LGBM_data/test_data_encoded.csv')
    #test_df = pd.read_parquet('data/test.parquet')

    customer_ID = test_df['customer_ID']
    #test_df = test_df.drop(['customer_ID','S_2'], axis=1)
    test_df = test_df.drop(['customer_ID'], axis=1)

    # # Impute categorical features with the most frequent value
    # cat_imputer = SimpleImputer(strategy='most_frequent') 
    # test_df[cat_features] = cat_imputer.fit_transform(test_df[cat_features])

    # # Get list of numerical features
    # num_features = [col for col in test_df.columns if col not in cat_features + ['customer_ID']]

    # # Impute numerical features with the mean
    # num_imputer = SimpleImputer(strategy='mean')
    # test_df[num_features] = num_imputer.fit_transform(test_df[num_features])

    # Encode categorical variables
    # label_encoders = {}
    # for cat_col in cat_features:
    #     label_encoders[cat_col] = LabelEncoder()
    #     test_df[cat_col] = label_encoders[cat_col].fit_transform(test_df[cat_col])

    # Predict
    test_pred = model.predict(test_df)

    # Output CSV
    output_df = pd.DataFrame({
        'customer_ID': customer_ID,
        'prediction': test_pred
    })
    output_df.to_csv('predictions.csv', index=False)
    print("Output Suc")

