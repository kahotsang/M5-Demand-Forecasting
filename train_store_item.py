"""
Train LSTM on store_item level
There are two types of data: endog (the historical time-series / lag-features) and exog (features other than lag-features)
exog data is prepared through DatasetGeneratorStoreItem class
model is trained using VanillaLSTMStoreItem class
"""

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from helper import construct_grouped_ts, compute_individual_ts
from LSTM_store_item import DatasetGeneratorStoreItem, VanillaLSTMStoreItem
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
dataset_generator = DatasetGeneratorStoreItem(calendar, sales_train, endog, prices_df) #prepare exog dataset generator
with open(MODEL_CHECKPOINT_DIR + 'store_item_dataset_generator.pkl', 'wb') as f:
    pkl.dump(dataset_generator, f)

#Model specification
vanilla_lstm = VanillaLSTMStoreItem(num_layers=2, num_units=[64,32], input_window=28, output_window=28, 
                                    encoder_exog_size=2, decoder_exog_size=30, lr=3e-4, dropout_rate=0, l2_regu=l2(1e-6),
                                    dataset_generator=dataset_generator)
model = vanilla_lstm.build_model()

#Training weights: following the evaluation metric for M5-Accuracy
scale = 1 / np.sqrt(np.square(endog.diff()).mean()) #denominator of RMSSE

train_dollar_sales = prices_df.loc[endog.index] * endog
loss_w = train_dollar_sales.tail(28).sum() * scale #Weighted loss similar to WRMSSE metric for M5-Accuracy
loss_w = loss_w / loss_w.sum() * endog.shape[1] #Scaled to mean 1

#Currently the data is not scaled to [0, 1]
train_st = int(endog.shape[0] - 365 * 2.5)
train_data = endog.iloc[train_st:, :] #training dataset for recent 2.5 yrs
train_gen = vanilla_lstm.prepare_training_dataset(train_data, loss_w, batch_size=256, sliding_freq=1) #Generator for training dataset
steps_per_epoch = int(endog.shape[1] / 256) * 2 #~ 200 steps per epoch

#Train model with checkpoints
num_iter = 100
for i in range(num_iter):
    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=1, verbose=1)
    if i in np.arange(80, 100):
        model.save_weights(MODEL_CHECKPOINT_DIR + 'store_item_model_weights_{}.h5'.format(i))

#Save the model design
json_string = model.to_json()
with open(MODEL_CHECKPOINT_DIR + 'store_item_model_design.pkl', 'wb') as f:
    pkl.dump(json_string, f)
    
#Estimate model resids----------------------------------------------------------------------------
#Construct ensemble models from checkpoints
models = []
for i in range(80, 100):
    model = model_from_json(json_string)
    model.load_weights(MODEL_CHECKPOINT_DIR + 'store_item_model_weights_{}.h5'.format(i))
    models.append(model)
K = len(models)

#Estimate resids using previous 112D forecasts
num_samples = 112
val_y = full_endog.iloc[-num_samples:,:] #previous 112D; on all agg levels

#Compute h-step ahead item_store level forecast for the previous 112D
#To reduce time: Assume that the h-step ahead resids are the same for all h        
base_forecasts_h = [pd.DataFrame(np.NaN, index=val_y.index, columns=endog.columns) for i in range(28)] #h-step ahead item_store forecast for val period
for t in np.arange(0, num_samples, 28):
    input_endog = endog.head(endog.shape[0]-num_samples+t).tail(28)
    val_idx = np.arange(input_endog.index[-1]+1, input_endog.index[-1]+29, 1)
    
    base_forecast = pd.DataFrame(0., index=val_idx, columns=endog.columns)
    for model in models:
        vanilla_lstm = VanillaLSTMStoreItem(input_window=28, output_window=28, model=model, dataset_generator=dataset_generator)
        forecast_df = vanilla_lstm.get_forecast(input_endog)
        base_forecast += np.array(forecast_df) / K
    
    for h, forecast_h in enumerate(base_forecasts_h):
        forecast_h.loc[val_idx] = base_forecast.values
        
#Only consider the forecast index in val_y
for h in range(28):
    base_forecasts_h[h] = base_forecasts_h[h][base_forecasts_h[h].index.isin(val_y.index)].astype(np.float32)

#Convert base_forecasts into full_forecast for resid on all agg levels
full_forecasts_h = []
for forecast_h in base_forecasts_h:
    full_forecast_h = pd.DataFrame(forecast_h.fillna(0).values * S, index=val_y.index, columns=val_y.columns)
    full_forecasts_h.append(full_forecast_h)
    
#Estimate of h-step ahead resid
resids_h = []
for h in range(28):
    resids_h.append(val_y - full_forecasts_h[h])
    
with open(LOGS_DIR + 'store_item_resid.pkl', 'wb') as f:
    pkl.dump(resids_h, f)
