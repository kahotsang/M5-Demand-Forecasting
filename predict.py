"""
Compute submission.csv using store_item LSTM and dept_store LSTM models.
"""

import numpy as np
import pandas as pd
import pickle as pkl
from helper import construct_grouped_ts, compute_individual_ts, construct_submission
from LSTM_dept_store import DatasetGeneratorDeptStore, VanillaLSTMDeptStore
from LSTM_store_item import DatasetGeneratorStoreItem, VanillaLSTMStoreItem
from keras.models import model_from_json
import scipy as sc

RAW_DATA_DIR = './raw_data/'
CLEAN_DATA_DIR = './processed_data/'
MODEL_CHECKPOINT_DIR = './model/'
LOGS_DIR = './logs/'
SUBMISSION_DIR = './submission/'

#Load data------------------------------------------------------------------------------------
sales_train = pd.read_csv(RAW_DATA_DIR + 'sales_train_evaluation.csv')
sample_submission = pd.read_csv(RAW_DATA_DIR + 'sample_submission.csv')
with open(CLEAN_DATA_DIR + 'st_list.pkl', 'rb') as f:
    st_list = pkl.load(f)
with open(CLEAN_DATA_DIR + 'endog.pkl', 'rb') as f:
    endog = pkl.load(f)
with open(CLEAN_DATA_DIR + 'S.pkl', 'rb') as f:
    S = pkl.load(f)
with open(CLEAN_DATA_DIR + 'full_ids.pkl', 'rb') as f:
    full_ids = pkl.load(f)
sales_train['st_idx'] = st_list

#Compute point forecast for store_item LSTM model---------------------------------------------
#load dataset_generator
with open(MODEL_CHECKPOINT_DIR + 'store_item_dataset_generator.pkl', 'rb') as f:
    store_item_dataset_generator = pkl.load(f)
    
#load model
with open(MODEL_CHECKPOINT_DIR + 'store_item_model_design.pkl', 'rb') as f:
    store_item_json_string = pkl.load(f)
store_item_models = []
for i in range(80, 100):
    model = model_from_json(store_item_json_string)
    model.load_weights(MODEL_CHECKPOINT_DIR + 'store_item_model_weights_{}.h5'.format(i))
    store_item_models.append(model)
K = len(store_item_models)

#Compute point forecast for next 28D
submission_index = np.arange(endog.index[-1]+1, endog.index[-1]+29) #timestamp for submission data (next 28D)

input_endog = endog.tail(28)
store_item_forecast = pd.DataFrame(0., index=submission_index, columns=endog.columns)
for model in store_item_models:
    vanilla_lstm = VanillaLSTMStoreItem(input_window=28, output_window=28, model=model, dataset_generator=store_item_dataset_generator)
    store_item_forecast += np.array(vanilla_lstm.get_forecast(input_endog)) / K
    
#Compute point forecast for dept_store LSTM model-------------------------------------------------------
#load dataset_generator
with open(MODEL_CHECKPOINT_DIR + 'dept_store_dataset_generator.pkl', 'rb') as f:
    dept_store_dataset_generator = pkl.load(f)
    
#load train_scale
with open(LOGS_DIR + 'dept_store_train_scale.pkl', 'rb') as f: #train_scale to multiply on the model agg_endog input
    dept_store_train_scale = pkl.load(f)
    
#load model
with open(MODEL_CHECKPOINT_DIR + 'dept_store_model_design.pkl', 'rb') as f:
    dept_store_json_string = pkl.load(f)
dept_store_models = []
for i in range(60, 80):
    model = model_from_json(dept_store_json_string)
    model.load_weights(MODEL_CHECKPOINT_DIR + 'dept_store_model_weights_{}.h5'.format(i))
    dept_store_models.append(model)
K = len(dept_store_models)

#Compute point forecast for next 28D, convert into store_item level using forecast ratio (previous 28D avg)
agg_endog, ratio_df = construct_grouped_ts(sales_train, endog, agg_1='dept_id', agg_2='store_id', drop_inactive=True, return_ratio=True)

input_endog = agg_endog.tail(28) * dept_store_train_scale
dept_store_agg_forecast = pd.DataFrame(0., index=submission_index, columns=agg_endog.columns)
for model in dept_store_models:
    vanilla_lstm = VanillaLSTMDeptStore(input_window=28, output_window=28, model=model, dataset_generator=dept_store_dataset_generator)
    dept_store_agg_forecast += np.array(vanilla_lstm.get_forecast(input_endog) / dept_store_train_scale) / K

#Forecast future ratio; using historical 28D avg
ratio_ts = ratio_df.drop('agg_idx', axis=1).T #Historical ratio of each store_item ts (as a proportion of the dept_store ts)
forecast_ratio = pd.concat([pd.DataFrame(ratio_ts.tail(28).mean()).T] * 28, axis=0)
forecast_ratio.index = submission_index
forecast_ratio_df = pd.concat([forecast_ratio.T, ratio_df[['agg_idx']]], axis=1)

#convert dept_store_agg_forecast into store_item level forecast, using forecast_ratio
dept_store_forecast = compute_individual_ts(dept_store_agg_forecast, forecast_ratio_df)

#convert individual point forecasts into prediction quantiles of ensemble model-----------------------------
#Load resids estimate
with open(LOGS_DIR + 'dept_store_resid.pkl', 'rb') as f:
    dept_store_resids = pkl.load(f)
    
with open(LOGS_DIR + 'store_item_resid.pkl', 'rb') as f:
    store_item_resids = pkl.load(f)

#Compute resid of simple ensemble of dept_store model & store_item model
resids = []
for h in range(28):
    idx = store_item_resids[h].index
    resid = pd.DataFrame((dept_store_resids[h].loc[idx].values + store_item_resids[h].values) / 2, index=idx, columns=store_item_resids[h].columns)
    resids.append(resid)

#Compute the confidence interval based on normality assumption on resids
norm = sc.stats.norm()
quantiles = np.array([0.005,0.025,0.165,0.25,0.5,0.75,0.835,0.975,0.995])
z_alpha = norm.ppf(quantiles) #Corresponding quantile for a standard normal rv

#Compute point forecast as simple ensemble of store_item_model & dept_store_model
point_forecast = (store_item_forecast.values + dept_store_forecast.values) / 2

#Convert point forecast into 12 agg levels
full_point_forecast = pd.DataFrame(point_forecast * S, index=submission_index, columns=np.arange(full_ids.shape[0])).astype(np.float32)

#Construct PI_forecasts (forecast for each quantile)
PI_forecasts = []
for i in range(len(quantiles)):
    quantile_forecast = pd.DataFrame(0., index=full_point_forecast.index, columns=full_point_forecast.columns).astype(np.float32) #Quantile forecast for the corresponding quantile
    base_timestep = np.arange(full_point_forecast.index[0], full_point_forecast.index[0] + full_point_forecast.shape[0], 28) #Indexes for 1-step ahead forecast
    for h in range(28):
        idx = base_timestep + h #Index for h-step ahead forecast
        resid_var = resids[h].var() #h-step forecast resid variance estimate
        quantile_forecast.loc[idx] = full_point_forecast.loc[idx].values + z_alpha[i] * np.sqrt(resid_var).values
    
    quantile_forecast[quantile_forecast < 0] = 0 #Clip at 0
    PI_forecasts.append(quantile_forecast)
    
submission = construct_submission(PI_forecasts, quantiles, full_ids, sample_submission, val_mode=False)
submission.to_csv(SUBMISSION_DIR + 'submission.csv', index=False)
