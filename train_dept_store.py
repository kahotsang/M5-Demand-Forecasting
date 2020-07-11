"""
Train LSTM on dept_store level
There are two types of data: endog (the historical time-series / lag-features) and exog (features other than lag-features)
exog data is prepared through DatasetGeneratorDeptStore class
model is trained using VanillaLSTMDeptStore class
"""
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from helper import construct_grouped_ts, compute_individual_ts
from LSTM_dept_store import DatasetGeneratorDeptStore, VanillaLSTMDeptStore
from keras.regularizers import l2
from keras.models import model_from_json

np.random.seed(99999)

RAW_DATA_DIR = './raw_data/'
CLEAN_DATA_DIR = './processed_data/'
MODEL_CHECKPOINT_DIR = './model/'
LOGS_DIR = './logs/'

#Load data------------------------------------------------------------------------------------
calendar = pd.read_csv(RAW_DATA_DIR + 'calendar.csv')
sales_train = pd.read_csv(RAW_DATA_DIR + 'sales_train_evaluation.csv')
with open(CLEAN_DATA_DIR + 'st_list.pkl', 'rb') as f:
    st_list = pkl.load(f)
with open(CLEAN_DATA_DIR + 'prices_df.pkl', 'rb') as f:
    prices_df = pkl.load(f)
with open(CLEAN_DATA_DIR + 'endog.pkl', 'rb') as f:
    endog = pkl.load(f)
with open(CLEAN_DATA_DIR + 'full_endog.pkl', 'rb') as f:
    full_endog = pkl.load(f)
with open(CLEAN_DATA_DIR + 'S.pkl', 'rb') as f:
    S = pkl.load(f)
sales_train['st_idx'] = st_list

#Model training----------------------------------------------------------------------------------
dataset_generator = DatasetGeneratorDeptStore(calendar, sales_train, endog, prices_df) #prepare exog dataset generator
with open(MODEL_CHECKPOINT_DIR + 'dept_store_dataset_generator.pkl', 'wb') as f:
    pkl.dump(dataset_generator, f)

#Model specification
vanilla_lstm = VanillaLSTMDeptStore(num_layers=2, num_units=[64,32], input_window=28, output_window=28, 
                                    encoder_exog_size=1, decoder_exog_size=27, lr=3e-4, dropout_rate=0, l2_regu=l2(1e-6),
                                    dataset_generator=dataset_generator)
model = vanilla_lstm.build_model()

#Prepare full training dataset (endog & exog), the agg_endog features are scaled into [0, 1]
agg_endog = dataset_generator.agg_endog.copy() #each col contains ts for dept_store level
train_scale = 1 / agg_endog.max() #Compute train_scale to convert agg_endog into [0, 1]
train_scale.to_pickle(LOGS_DIR + 'dept_store_train_scale.pkl') #Save train_scale for future conversion

agg_loss_w = pd.Series(1, index=agg_endog.columns) #Unweighted loss
train_X, train_Y, train_w = vanilla_lstm.prepare_training_dataset(agg_endog * train_scale, agg_loss_w, sliding_freq=1)

#Train model with checkpoints
num_iter = 80
for i in range(num_iter):
    model.fit(train_X, train_Y, sample_weight=train_w, batch_size=256, epochs=1, shuffle=True)
    if i in np.arange(60, 80):
        model.save_weights(MODEL_CHECKPOINT_DIR + 'dept_store_model_weights_{}.h5'.format(i))

#Save the model design
json_string = model.to_json()
with open(MODEL_CHECKPOINT_DIR + 'dept_store_model_design.pkl', 'wb') as f:
    pkl.dump(json_string, f)

#Estimate model resids----------------------------------------------------------------------------
#Construct ensemble models from checkpoints
models = []
for i in range(60, 80):
    model = model_from_json(json_string)
    model.load_weights(MODEL_CHECKPOINT_DIR + 'dept_store_model_weights_{}.h5'.format(i))
    models.append(model)
K = len(models)

#Estimate resids using previous 112D forecasts
num_samples = 112
val_y = full_endog.iloc[-num_samples:,:] #previous 112D; on all agg levels

#Compute forecast ratio for previous 112D; forecast ratio is computed by rolling_avg of previous 28D ratio
#forecast ratio is used to convert dept_store ts into item_store ts
ratio_df = dataset_generator.ratio_df.copy() #Historical ratio of each store_item ts (as a proportion of the dept_store ts)
ratio_ts = ratio_df.drop('agg_idx', axis=1).T

forecast_ratio_h = [pd.DataFrame(np.NaN, index=val_y.index, columns=endog.columns) for h in range(28)] #h-step ahead forecast ratio for val period
for h in range(28):
    forecast_ratio_h[h] = ratio_ts.shift(h+1).rolling(28, min_periods=1).mean().loc[val_y.index]

#Compute h-step ahead dept_store level forecast for the previous 112D
base_forecasts_h = [pd.DataFrame(np.NaN, index=val_y.index, columns=agg_endog.columns) for i in range(28)] #h-step ahead dept_store_forecast for val period
for t in np.arange(-27, num_samples, 1):
    input_endog = agg_endog.head(agg_endog.shape[0]-num_samples+t).tail(28) * train_scale
    val_idx = np.arange(input_endog.index[-1]+1, input_endog.index[-1]+29, 1)
    
    base_forecast = pd.DataFrame(0., index=val_idx, columns=agg_endog.columns)
    for model in models:
        vanilla_lstm = VanillaLSTMDeptStore(input_window=28, output_window=28, model=model, dataset_generator=dataset_generator)
        forecast_df = vanilla_lstm.get_forecast(input_endog) / train_scale
        base_forecast += np.array(forecast_df) / K
    
    for h, forecast_h in enumerate(base_forecasts_h):
        idx = base_forecast.index[h]
        forecast_h.loc[idx] = base_forecast.loc[idx].values
        
#Only consider the forecast index in val_y
for h in range(28):
    base_forecasts_h[h] = base_forecasts_h[h][base_forecasts_h[h].index.isin(val_y.index)].astype(np.float32)

#Convert base_forecasts into full_forecast for resid on all agg_levels
full_forecasts_h = []
for h, forecast_h in enumerate(base_forecasts_h):
    forecast_ratio_df = pd.concat([forecast_ratio_h[h].T, ratio_df[['agg_idx']]], axis=1) #h-step ahead forecast ratio
    individual_forecast_h = compute_individual_ts(forecast_h, forecast_ratio_df).astype(np.float32) #forecast on store_item_level
    full_forecast_h = pd.DataFrame(individual_forecast_h.fillna(0).values * S, index=val_y.index, columns=val_y.columns) #forecast on all agg_levels
    full_forecasts_h.append(full_forecast_h)
    
#Estimate of h-step ahead resid
resids_h = []
for h in range(28):
    resids_h.append(val_y - full_forecasts_h[h])
    
with open(LOGS_DIR + 'dept_store_resid.pkl', 'wb') as f:
    pkl.dump(resids_h, f)
