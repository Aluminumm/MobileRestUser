# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

uid_train = pd.read_csv('uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'))
sms_train = pd.read_csv('sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'))
wa_train = pd.read_csv('wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'))

voice_test = pd.read_csv('voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'))
sms_test = pd.read_csv('sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'))
wa_test = pd.read_csv('wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'))

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('uid_test_b.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)


#通话记录
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()
voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)
voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)
voice['voice_dura'] = (voice['end_time'] - voice['start_time'])
voice['voice_dura_hour'] = (voice['voice_dura']/ 1000000).astype('int')
voice_in_out['in_larger_out'] = voice_in_out['voice_in_out_1'] - voice_in_out['voice_in_out_0']

voice_in_larger_out = voice.groupby(['uid','opp_num','in_out'])['uid'].count().unstack().add_prefix('voice_in_vs_out_').reset_index().fillna(0)
voice_in_larger_out['in_larger_out'] = voice_in_larger_out['voice_in_vs_out_0'] - voice_in_larger_out['voice_in_vs_out_1']
voice_in_larger_out = voice_in_larger_out.groupby(['uid'])['in_larger_out'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_opp_in&out_').reset_index().fillna(0)

'''
voice_opp_dura = voice.groupby(['uid'])['voice_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_opp_dura_').reset_index()
voice_dura_hour = voice.groupby(['uid'])['voice_dura_hour'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_opp_dura_').reset_index()
voice_opp_least = voice.groupby(['uid'])['start_time'].agg(['max','min','median']).add_prefix('voice_opp_start_').reset_index().fillna(0)
'''

voice_opp_head = voice.groupby(['uid'])['opp_head'].agg(
    {'unique_count': lambda x: len(pd.unique(x)), 'max_count': lambda x: x.value_counts().index[0]}).add_prefix('voice_opp_head_').reset_index()
def opp_head_size(x):
    return  str(x).__len__()
voice_opp_head_x = voice.opp_head.apply(opp_head_size)
voice['opp_head_size']=voice_opp_head_x
voice_opp_head_y = voice.groupby(['uid','opp_head_size'])['uid'].count().unstack().add_prefix('voice_opp_head_').reset_index().fillna(0)
voice_opp_head = pd.merge(voice_opp_head, voice_opp_head_y, how='left', on='uid')

'''
voice_opp_date = voice.groupby(['uid','start_date'])['uid'].count().unstack().add_prefix('voice_start_date_').reset_index().fillna(0)
voice['start_date'] = (voice['start_time'] / 1000000).astype('int')
voice_date = voice.groupby(['uid','start_date'])['start_date'].agg(['max','min','median']).add_prefix('voice_start_date_').reset_index()
'''

#短信记录
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()
sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)
sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)
sms_in_out['in_larger_out'] = sms_in_out['sms_in_out_1'] - sms_in_out['sms_in_out_0']
#sms_opp_least = sms.groupby(['uid'])['start_time'].agg(['max','min','median']).add_prefix('sms_opp_start_').reset_index().fillna(0)

sms['in_larger_out'] = sms_in_out['in_larger_out']
sms_in_larger_out = sms.groupby(['uid'])['in_larger_out'].agg(['max','min','median']).add_prefix('sms_opp_in&out_').reset_index().fillna(0)

sms_in_larger_out = sms.groupby(['uid','opp_num','in_out'])['uid'].count().unstack().add_prefix('sms_in_vs_out_').reset_index().fillna(0)
sms_in_larger_out['in_larger_out'] = sms_in_larger_out['sms_in_vs_out_0'] - sms_in_larger_out['sms_in_vs_out_1']
sms_in_larger_out = sms_in_larger_out.groupby(['uid'])['in_larger_out'].agg(['max','min','median']).add_prefix('sms_opp_in&out_').reset_index().fillna(0)

sms_opp_head = sms.groupby(['uid'])['opp_head'].agg(
    {'unique_count': lambda x: len(pd.unique(x)), 'max_count': lambda x: x.value_counts().index[0]}).add_prefix('sms_opp_head_').reset_index()
def opp_head_size(x):
    return  str(x).__len__()

'''
sms['start_date'] = (sms['start_time'] / 1000000).astype('int')
sms['start_hour'] = ((sms['start_time'] / 10000).astype('int')) % 100
sms_hour = sms.groupby(['uid'])['start_hour'].agg(['std','max','min','median','mean']).add_prefix('sms_start_hour_').reset_index()
sms_date = sms.groupby(['uid','start_date'])['start_date'].agg(['std','max','min','median','mean']).add_prefix('sms_start_date_').reset_index()
sms_opp_date = sms.groupby(['uid','start_date'])['uid'].count().unstack().add_prefix('sms_start_date_').reset_index().fillna(0)
'''

#网站\APP记录
wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()
up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()
down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()
wa_date = wa.groupby(['uid'])['date'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_date_').reset_index()
wa_type = wa.groupby(['uid','wa_type'])['uid'].count().unstack().add_prefix('wa_type_').reset_index().fillna(0)

'''
wa_date_num = wa.groupby(['uid'])['date'].agg(['max','min','mean']).add_prefix('wa_date_num_').reset_index()
wa['dura_per_cnt'] = wa['visit_dura']/ wa['visit_cnt']
wa_dura_per_cnt = wa.groupby(['uid'])['dura_per_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()
wa['up_larger_down'] = wa['up_flow'] - wa['down_flow']
wa_up_larger_down = wa.groupby(['uid'])['up_larger_down'].agg(['std','max','min']).add_prefix('wa_down_flow_').reset_index()
'''

feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_opp_least,voice_in_larger_out,
           sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,sms_in_larger_out,
           wa_name,visit_cnt,visit_dura,up_flow,down_flow,wa_date,wa_type]
train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')
test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')

train_feature.to_csv('train_featureV1.csv', index=None)
test_feature.to_csv('test_featureV1.csv', index=None)
