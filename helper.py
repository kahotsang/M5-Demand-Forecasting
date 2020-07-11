"""
General helper functions
"""
import numpy as np
import pandas as pd

def remove_inactive_periods(endog, st_idx):
    """
    Supporting function for construct_grouped_ts if drop_inactive = True
    
    Input: endog: each column represents a time series. Index represents the timestamp.
    st_idx (np.array) gives the first non_zero index of each time series.
    Return endog with the beginning zeros (index < st_idx) set as np.NaN
    """
    
    endog = endog.astype(np.float32).copy()
    
    for i, col in enumerate(endog.columns):
        endog.loc[endog.index < st_idx[i], col] = np.NaN
    
    return(endog)

def construct_grouped_ts(sales_train, endog, agg_1=None, agg_2=None, drop_inactive=False, return_ratio=False, return_ids=False):
    """
    Input: endog represents the time series where each column represents a ts of store_item pair, ordered in the same way as in sales_train. Shape = [n_obs, n_individual_ts]
    Index of endog must match with actual timestamp if drop_inactive = True.
    Return the grouped_ts which is aggregated sum according to agg_1 and agg_2. Shape = [n_obs, n_groups]
    
    If drop_inactive = True: drop the beginning zeros (index < st_idx) of grouped_ts, as indicated by st_idx in sales_train.
    
    If return_ratio = True: Also return the ratio_df, where each row represents a store_item ts (in the same order as sales_train), and column represents timestamp in endog. 
    Each element is the ratio of store_item sales / agg_sales. ratio_df can be passed to compute_individual_ts() for computing store_item sales.
    Shape = [n_individual_ts, n_obs + 1]
    
    If return_ids = True: Also return the ids which specifies the name of each time series
    """
    
    df = pd.concat([sales_train[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'st_idx']], endog.T], axis=1)
    
    if (agg_1):
        if (agg_2):
            tmp = df.groupby([agg_1, agg_2])[endog.index].sum().reset_index()
            
            if (agg_1 == 'item_id' and agg_2 == 'store_id'):
                ids = tmp[agg_1] + '_' + tmp[agg_2]
            else:
                ids = tmp[agg_2] + '_' + tmp[agg_1]
            grouped_ts = tmp[endog.index].T
            st_idx = np.array(df.groupby([agg_1, agg_2])['st_idx'].min())
        else:
            tmp = df.groupby([agg_1])[endog.index].sum().reset_index()
            
            ids = tmp[agg_1] + '_X' 
            grouped_ts = tmp[endog.index].T
            st_idx = np.array(df.groupby([agg_1])['st_idx'].min())
    else:
        ids = pd.Series('Total_X')
        grouped_ts = pd.DataFrame(endog.sum(axis=1))
        st_idx = np.array([df['st_idx'].min()])
        
    if drop_inactive:
        grouped_ts = remove_inactive_periods(grouped_ts, st_idx)
        
    grouped_ts.index = endog.index #Correct the dtypes of index of grouped_ts
    
    if return_ratio:
        if (agg_1):
            if (agg_2):
                agg_ts = df.groupby([agg_1, agg_2])[endog.index].sum()
            else:
                agg_ts = df.groupby([agg_1])[endog.index].sum()
            
            agg_cols = ['agg_{}'.format(col) for col in endog.index]
            agg_ts.columns = agg_cols
            agg_ts.reset_index(inplace=True)
            agg_ts['agg_idx'] = np.arange(agg_ts.shape[0])
            
            if (agg_2):
                merged_df = pd.merge(df, agg_ts, on=[agg_1, agg_2], how='left')
            else:
                merged_df = pd.merge(df, agg_ts, on=[agg_1], how='left')
            
            ratio_df = pd.DataFrame(np.array(merged_df[endog.index]) / np.array(merged_df[agg_cols]), index=sales_train.index, columns=endog.index)
            ratio_df['agg_idx'] = merged_df['agg_idx']
        
        if return_ids:
            return([grouped_ts, ratio_df, ids])
        else:
            return([grouped_ts, ratio_df])
    else:
        if return_ids:
            return([grouped_ts, ids])
        else:
            return(grouped_ts)
    
def compute_individual_ts(grouped_ts, ratio_df):
    """
    Input: the grouped_ts in aggregate level.
    ratio_df for converting grouped_ts into individual ts with its ratio at each timestamp, where ratio_df['agg_idx'] represents the associated col_idx in grouped_ts.
    grouped_ts shaped [n_obs, n_groups]; ratio_df shaped [n_individual_ts, n_obs + 1].
    
    Return individual_ts shaped [n_obs, n_individual_ts].
    """
    
    individual_ts = ratio_df.copy().drop('agg_idx', axis=1)
    for i, col in enumerate(grouped_ts.columns):
        individual_ts.loc[ratio_df['agg_idx'] == i] = grouped_ts[col] * individual_ts.loc[ratio_df['agg_idx'] == i]
        
    return(individual_ts.T)

def construct_submission(PI_forecasts, quantiles, full_ids, sample_submission, val_mode=True):
    """
    Convert PI_forecasts into submission format.
    """
    F_cols = ['F{}'.format(i) for i in range(1, 29)]
    PIs_submission = []
    for i, q in enumerate(quantiles):
        quantile_submission = pd.DataFrame(columns=['id'] + F_cols)
        
        if val_mode:
            quantile_submission['id'] = full_ids + '_{:.3f}_validation'.format(q)
        else:
            quantile_submission['id'] = full_ids + '_{:.3f}_evaluation'.format(q)
            
        quantile_submission[F_cols] = PI_forecasts[i].T
        PIs_submission.append(quantile_submission)

    PIs_submission = pd.concat(PIs_submission, axis=0)
    
    N = int(sample_submission.shape[0]/2)
    if val_mode:
        val_submission = pd.merge(sample_submission.head(N)[['id']], PIs_submission, on='id', how='left')
        eval_submission = sample_submission.tail(N)
    else:
        val_submission = sample_submission.head(N)
        eval_submission = pd.merge(sample_submission.tail(N)[['id']], PIs_submission, on='id', how='left')
        
    submission = pd.concat([val_submission, eval_submission], axis=0)
    
    return(submission)