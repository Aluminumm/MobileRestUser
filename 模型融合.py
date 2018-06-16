# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

lgb_res=pd.read_csv('result.csv', header=None, names=('uid', 'prob1'))
xgb_res=pd.read_csv('xgBoost.csv', header=None, names=('uid', 'prob2'))

result = lgb_res.merge(xgb_res,on='uid')
result['label']=(result['prob1']+result['prob2'])/2
result=result.sort_values(by='label',ascending=False)

result.label=result.label.map(lambda x: 1 if x>0.28 else 0)
result.label=result.label.map(lambda x: int(x))
res = pd.DataFrame({'uid':lgb_res.uid,'label':result.label})
res = res.sort_values(by='label',ascending=False)
res.to_csv('final.csv',index=False,header=False,sep=',',columns=['uid','label'])

