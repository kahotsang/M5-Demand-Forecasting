"""
Data preparation for M5 Competition

Data for store_item level:
endog: each col represents a ts of item_store pair, must be ordered as the same way as in sales_train (by col). The beginning zeros of endog are replaced by np.NaN
st_list: the first non-zero index of each ts. Ordered in the same way as endog
prices_df: the price of each item_store pair. Indexed in the same way as endog

Data for all agg_levels: (12 agg_levels as required)
S: Structural matrix which can convert data from item_store level to all agg_levels
full_endog: each col represents a ts of all time-series (i.e. including all agg_levels), with beginning zeros replaced by np.NaN
full_ids: specify the name of each time-series of full_endog
"""

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
from helper import construct_grouped_ts, compute_individual_ts

RAW_DATA_DIR = './raw_data/'
CLEAN_DATA_DIR = './processed_data/'

#Load raw data
calendar = pd.read_csv(RAW_DATA_DIR + 'calendar.csv')
sales_train = pd.read_csv(RAW_DATA_DIR + 'sales_train_evaluation.csv')
sell_prices = pd.read_csv(RAW_DATA_DIR + 'sell_prices.csv')

#Create endog: An alternative data representation of time series data
ts_cols = ['d_{}'.format(i) for i in range(1, 1942)]
endog = sales_train[ts_cols].T.astype(np.float32)
endog.index = np.arange(endog.shape[0])

#List on the first non-zero day for each ts: for future processing of endog
st_list = []
for i in range(sales_train.shape[0]):
    st_list.append(np.min(np.where(sales_train.loc[i, ts_cols] > 0)))
st_list = np.array(st_list)

#Transform the endog with beginning zeros to be NaN
for i, col in enumerate(endog.columns):
    endog.loc[endog.index < st_list[i], col] = np.NaN

#Construct the price table for each store_item. NaN for items before active (before st_list)
prices_df = pd.DataFrame(index=np.arange(calendar.shape[0]))
keys_to_store_item = sales_train[['store_id', 'item_id']] #for mapping from endog to sales_train
for i in keys_to_store_item.index:
    store_id, item_id = keys_to_store_item.loc[i,:]
    
    sell_prices_subset = sell_prices.head(1000)
    prices_store_item = sell_prices_subset.loc[(sell_prices_subset.store_id == store_id) & (sell_prices_subset.item_id == item_id), ['wm_yr_wk', 'sell_price']]
    prices_df[i] = pd.merge(calendar[['wm_yr_wk']], prices_store_item, on='wm_yr_wk', how='left', copy=False)['sell_price']
    
    sell_prices = sell_prices.tail(sell_prices.shape[0] - prices_store_item.shape[0])
prices_df = prices_df.astype(np.float32)

#Prepare S & full_endog---------------------------------------------------------------
#Index the grouped level
sales_train['st_idx'] = st_list
sales_train['all_id'] = 'all'
agg_cols = [
    ['item_id', 'state_id'],
    ['dept_id', 'store_id'],
    ['dept_id', 'state_id'],
    ['cat_id', 'store_id'],
    ['cat_id', 'state_id']
]
for agg_col in agg_cols:
    key = '_'.join(agg_col)
    sales_train[key] = sales_train[agg_col[0]] + '_' + sales_train[agg_col[1]]

#Prepare S
agg_levels = ['id', 'item_id_state_id', 'item_id', 'dept_id_store_id', 'cat_id_store_id', 
              'dept_id_state_id', 'cat_id_state_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'all_id']
S = OneHotEncoder(drop=None, sparse=True).fit_transform(sales_train[agg_levels]).astype(np.int8)

#Prepare full_endog: Need to replace beginning zeros as np.NaN
agg_levels = [
        ['item_id', 'store_id'],
        ['item_id', 'state_id'],
        ['item_id', None],
        ['dept_id', 'store_id'],
        ['cat_id', 'store_id'],
        ['dept_id', 'state_id'],
        ['cat_id', 'state_id'],
        ['dept_id', None],
        ['cat_id', None],
        ['store_id', None],
        ['state_id', None],
        [None, None]
             ]

full_endog = pd.DataFrame()
full_ids = pd.Series()
for agg_level in agg_levels:
    agg_endog, agg_ids = construct_grouped_ts(sales_train, endog, agg_1=agg_level[0], agg_2=agg_level[1], drop_inactive=True, return_ids=True)
    full_endog = pd.concat([full_endog, agg_endog], axis=1)
    full_ids = pd.concat([full_ids, agg_ids])
full_endog.columns = np.arange(full_endog.shape[1])
full_ids.index = np.arange(full_ids.shape[0])

#Save processed_data------------------------------------------------
endog.to_pickle(CLEAN_DATA_DIR + 'endog.pkl')
with open(CLEAN_DATA_DIR + 'st_list.pkl', 'wb') as f:
    pkl.dump(st_list, f)
prices_df.to_pickle(CLEAN_DATA_DIR + 'prices_df.pkl')

full_endog.to_pickle(CLEAN_DATA_DIR + 'full_endog.pkl')
full_ids.to_pickle(CLEAN_DATA_DIR + 'full_ids.pkl')
with open(CLEAN_DATA_DIR + 'S.pkl', 'wb') as f:
    pkl.dump(S, f)
