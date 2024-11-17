import numpy as np
import pandas as pd
import contextlib
import logging
import os
import json

from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

lgbm_params_path = os.path.join(os.getcwd(), '../best_parameters', 'lgbm_params.json')
catboost_params_path = os.path.join(os.getcwd(), '../best_parameters', 'catboost_params.json')

trainig_data_path = os.path.join(os.getcwd(), '../training_data', 'training_data.tsv')
processed_data_path = os.path.join(os.getcwd(), '../temporary_data', 'processed_data.tsv')
output_predictions_path = os.path.join(os.getcwd(), '../../output_data', 'payments_main.tsv')

with open(lgbm_params_path, 'r') as f:
    lgbm_params = json.load(f)

with open(catboost_params_path, 'r') as f:
    catboost_params = json.load(f)

logging.basicConfig(level=logging.CRITICAL)

with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
    lgbm_model = LGBMClassifier(
        n_estimators=int(lgbm_params['n_estimators']),
        max_depth=int(lgbm_params['max_depth']),
        min_samples_split=int(lgbm_params['min_samples_split']),
        min_samples_leaf=int(lgbm_params['min_samples_leaf']),
        max_features=lgbm_params['max_features'],
        bootstrap=bool(lgbm_params['bootstrap']),
        oob_score=bool(lgbm_params['oob_score']) if lgbm_params.get('bootstrap', False) else False,
        random_state=42,
        verbose=0,
    )

    catboost_model = CatBoostClassifier(
        learning_rate=catboost_params['learning_rate'],
        depth=int(catboost_params['depth']),
        iterations=int(catboost_params['iterations']),
        min_data_in_leaf=int(catboost_params['min_data_in_leaf']),
        max_bin=int(catboost_params['max_bin']),
        random_state=42,
        verbose=0,
    )

    df = pd.read_csv(trainig_data_path, sep='\t')

    X = df.drop(columns=['target'])
    y = df['target']
    X, y = shuffle(X, y, random_state=42)

    logger.info("Initializing StackingClassifier model...")
    stacking_model = StackingClassifier(
        estimators=[('lgbm', lgbm_model), ('catboost', catboost_model)],
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=3,
        stack_method='predict_proba',
        verbose=0,
        passthrough=True
    )

    logger.info("Starting cross-validation...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        logger.info(f"Starting fold {fold_idx + 1}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        logger.info(f"Fitting model on fold {fold_idx + 1}...")
        stacking_model.fit(X_train, y_train)
        y_pred = stacking_model.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_val, y_pred, average='macro'))
        f1_scores.append(f1_score(y_val, y_pred, average='macro'))

        logger.info(f"Finished fold {fold_idx + 1}.")

    df_processed = pd.read_csv(processed_data_path, sep='\t')
    X_test = df_processed.drop(columns=['id'])
    ids = df_processed['id']

    logger.info("Predicting on your data...")
    predicted_classes = stacking_model.predict(X_test)

    output_df = pd.DataFrame({'id': ids, 'predicted_class': predicted_classes})
    output_df.to_csv(output_predictions_path, sep='\t', index=False, header=False)

    logger.info(f"Predicted classes saved to {output_predictions_path}")