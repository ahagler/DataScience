import pandas as pd
import xgboost as xgb

best_features = ['isFraud', 'card2', 'addr1', 'C4', 'C10', 'V87', 'card5', 'D4', 'C11', 'TransactionAmt', 'D1', 'C8', 'C6', 'D2', 'C13', 'C14', 'TransactionDT', 'D15', 'C1', 'C2', 'card1']
def clean_data(data):
  new_data = data[best_features]
  tot = new_data.shape[0]
  for feature in new_data.columns:
    n = train_txn[feature].isnull().sum()
    if n < .1*tot:
      new_data = new_data[new_data[feature].notna()]
    else:
      new_data[feature] = new_data[feature].fillna(new_data[feature].mean())
  y = new_data['isFraud']
  del new_data['isFraud']
  return new_data, y

def build_model(X, y):
	model = xgb.XGBClassifier(
	    n_estimators=500,
	    max_depth=9,
	    learning_rate=0.05,
	    subsample=0.9,
	    colsample_bytree=0.9,
	    missing=-999,
	    random_state=2
	)
	model.fit(X.values, y.values, eval_metric="auc")
	return model