# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance

train = pd.read_csv('train_featureV1.csv')
test = pd.read_csv('test_featureV1.csv')

y = train.label
X = train.drop(['uid', 'label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
y_test = y_test.reset_index(drop=True)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 8,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'slient': 1,
    'eta': 0.1,
    'seed': 100,
    'nthread': 4,
}

plst = params.items()

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

dtest = xgb.DMatrix(X_test)
res = model.predict(dtest)


dy_test = test.drop(['uid'],axis=1)
dy_test = xgb.DMatrix(dy_test)
pred = model.predict(dy_test)
pre = pd.DataFrame(pred,columns=['lable0','label1'])

res = pd.DataFrame({'uid':test.uid,'label':pre.label1})
res = res.sort_values(by='label',ascending=False)
#res.label=res.label.map(lambda x: 1 if x>=0.28 else 0)
#res.label = res.label.map(lambda x: int(x))
res.to_csv('xgBoost.csv',index=False,header=False,sep=',',columns=['uid','label'])
