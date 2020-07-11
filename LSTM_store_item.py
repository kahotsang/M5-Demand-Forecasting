"""
Class implementation for training LSTM on store_item level
There are two types of data: endog (the historical time-series / lag-features) and exog (features other than lag-features)
exog data is prepared through DatasetGeneratorStoreItem class
endog data is prepared through endog (on store_item level)

model is trained using VanillaLSTMStoreItem class,
and is trained unconditionally (i.e. the decoder does not take any forecast value as input).
"""

import numpy as np
import pandas as pd
from helper import construct_grouped_ts, compute_individual_ts
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Concatenate, Embedding
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf

class DatasetGeneratorStoreItem():
    """
    Provide exog dataset generation method (load_) for training and inference of VanillaLSTMStoreItem
    """
    def __init__(self, calendar, sales_train, endog, prices_df):
        self.calendar = calendar.copy()
        self.sales_train = sales_train.copy()
        self.endog = endog.copy()
        self.prices_df = prices_df.copy()
        
        self.prepare_dataset_encoder()
    
    def prepare_dataset_encoder(self):
        """
        Prepare dataset and encoder for load methods
        """
        calendar, sales_train, endog, prices_df = self.calendar, self.sales_train, self.endog, self.prices_df
        
        #Prepare exog dataset ---------------------------------------------------------------
        #Prepare calendar_exog: event_type & wday on a date
        calendar_exog = pd.DataFrame(index=calendar.index)
        for event_type in ['Sporting', 'Cultural', 'National', 'Religious']:
            calendar_exog['is_{}'.format(event_type)] = np.where((calendar.loc[calendar_exog.index, ['event_type_1', 'event_type_2']] == event_type).any(axis=1), 1, 0)
        wday_encoder = OneHotEncoder(drop='first', sparse=False) #drop Sat.
        wday_df = pd.DataFrame(wday_encoder.fit_transform(calendar.loc[calendar_exog.index, ['wday']]), columns=['w7'] + ['w{}'.format(i) for i in range(1,6)])
        calendar_exog = pd.concat([calendar_exog, wday_df], axis=1)
        
        #Prepare snap_exog: if there is snap event on that date & store_item ts
        snap_exog = pd.DataFrame(0., index=calendar.index, columns=endog.columns)
        for state_id in ['CA', 'TX', 'WI']:
            state_idx = sales_train[sales_train.state_id == state_id].index
            snap_exog[state_idx] = np.repeat(np.array(calendar.loc[snap_exog.index, ['snap_{}'.format(state_id)]]), repeats = state_idx.shape[0], axis=1)
        
        #Prepare price diff & price discount on that date & store_item ts
        price_discount = prices_df / prices_df.max() #scaled to [0, 1]
            
        price_diff_28 = np.log(prices_df).diff(28)
        price_diff_28_mean, price_diff_28_std = pd.Series(price_diff_28.values.reshape((-1,))).mean(), pd.Series(price_diff_28.values.reshape((-1,))).std()
        price_diff_28 = (price_diff_28 - price_diff_28_mean) / (price_diff_28_std) #normalized
    
        #agg_sales_features: item level and dept_store level ts
        item_endog, item_ratio = construct_grouped_ts(sales_train, endog, agg_1='item_id', agg_2=None, drop_inactive=True, return_ratio=True)
        item_endog = item_endog / item_endog.max() #scaled to [0, 1]
        item_agg_idx = item_ratio.agg_idx #Index for referencing store_item level ts with a item level ts
        
        dept_store_endog, dept_store_ratio = construct_grouped_ts(sales_train, endog, agg_1='dept_id', agg_2='store_id', drop_inactive=True, return_ratio=True)
        dept_store_endog = dept_store_endog / dept_store_endog.max() #scaled to (0,1)
        dept_store_agg_idx = dept_store_ratio.agg_idx #Index for referencing store_item level ts with a dept_store level ts
        
        self.calendar_exog = calendar_exog
        self.snap_exog = snap_exog
        self.price_discount = price_discount
        self.price_diff_28 = price_diff_28
        self.item_endog = item_endog
        self.item_agg_idx = item_agg_idx
        self.dept_store_endog = dept_store_endog
        self.dept_store_agg_idx = dept_store_agg_idx
        
        #Prepare encoder ----------------------------------------------------------------------
        #encoder for dept_store id of a store_item ts
        dept_store_encoder = OneHotEncoder(drop='first', sparse=False).fit(sales_train[['dept_id', 'store_id']])
        
        #encoder for item id of a store_item ts
        item_encoder = LabelEncoder().fit(sales_train['item_id'])
        
        #encoder for event_name
        calendar['event_name_1'].fillna('missing', inplace=True)
        event_encoder = LabelEncoder().fit(calendar['event_name_1'])
        
        self.dept_store_encoder = dept_store_encoder
        self.item_encoder = item_encoder
        self.event_encoder = event_encoder
        
    def load_encoder_exog(self, exog_index, exog_cols):
        """
        Load encoder_exog at timestamp (exog_index), store_items (exog_columns).
        encoder_exog with shape (n_ts, n_timestamp, n_features)
        """
        item_exog_cols = np.array(self.item_agg_idx.loc[exog_cols]) #reference to the associated item level ts
        item_exog = np.array(self.item_endog.loc[exog_index, item_exog_cols].T).reshape((exog_cols.shape[0], -1, 1))
        
        dept_store_exog_cols = np.array(self.dept_store_agg_idx.loc[exog_cols]) #reference to the associated dept_store level ts
        dept_store_exog = np.array(self.dept_store_endog.loc[exog_index, dept_store_exog_cols].T).reshape((exog_cols.shape[0], -1, 1))
        
        encoder_exog = np.concatenate([item_exog, dept_store_exog], axis=-1)
        return(encoder_exog.astype(np.float32))
    
    def load_decoder_exog(self, exog_index, exog_cols):
        """
        Load decoder_exog at timestamp (exog_index), store_items (exog_columns).
        decoder_exog with shape (n_ts, n_timestamp, n_features)
        """
        decoder_exog = np.repeat(np.array(self.calendar_exog.loc[exog_index]).reshape((1,exog_index.shape[0],-1)), repeats=exog_cols.shape[0], axis=0)
        decoder_exog = np.concatenate([decoder_exog, np.array(self.snap_exog.loc[exog_index, exog_cols].T).reshape((exog_cols.shape[0], -1, 1))], axis=-1)
        decoder_exog = np.concatenate([decoder_exog, np.array(self.price_diff_28.loc[exog_index, exog_cols].T).reshape((exog_cols.shape[0], -1, 1))], axis=-1)
        decoder_exog = np.concatenate([decoder_exog, np.array(self.price_discount.loc[exog_index, exog_cols].T).reshape((exog_cols.shape[0], -1, 1))], axis=-1)
        
        #Append with id features: dept_store_id with one-hot encoded
        id_features =  self.dept_store_encoder.transform(self.sales_train.loc[exog_cols, ['dept_id', 'store_id']])
        id_features = np.repeat(np.expand_dims(id_features, axis=1), repeats=exog_index.shape[0], axis=1)
        decoder_exog = np.concatenate([decoder_exog, id_features], axis=-1)
        
        #Append with manual lag features (long-term)
        idx_ed = exog_index.min()
        lag_360_mean = np.repeat(np.array(self.endog.loc[(idx_ed-360):(idx_ed-1), exog_cols].mean()).reshape((-1,1,1)), repeats=exog_index.shape[0], axis=1)
        lag_360_std = np.repeat(np.array(self.endog.loc[(idx_ed-360):(idx_ed-1), exog_cols].std()).reshape((-1,1,1)), repeats=exog_index.shape[0], axis=1)
        lag_features = np.concatenate([lag_360_mean, lag_360_std], axis=-1)
        
        decoder_exog = np.concatenate([decoder_exog, lag_features], axis=-1)
        return(decoder_exog.astype(np.float32))
    
    def load_item_id(self, exog_index, exog_cols):
        """
        Load item_id at timestamp (exog_index), store_items (exog_columns).
        item_id with shape (n_ts, n_timestamp).
        """
        
        item_id = self.item_encoder.transform(self.sales_train.loc[exog_cols, 'item_id']).reshape((-1,1))
        item_id = np.repeat(item_id, repeats=exog_index.shape[0], axis=1)
        
        return(item_id)
    
    def load_event_name(self, exog_index, exog_cols):
        """
        Load event_id at timestamp (exog_index), store_items (exog_columns).
        event_name with shape (n_ts, n_timestamp).
        """
        
        event_name = self.event_encoder.transform(self.calendar.loc[exog_index, 'event_name_1']).reshape((1,-1))
        event_name = np.repeat(event_name, repeats=exog_cols.shape[0], axis=0)
        
        return(event_name)

def loss_function(y_true, y_pred):
        """
        Custom loss function for VanillaLSTMStoreItem model (zero-inflated poisson loss)
        y_true / y_pred: shape = (batch_size, num_timestamp, 2); last dimension: (z, S).
        z: values of y; S: if y == 0.
        """
        gamma = 1 #control the balance between classification and regression (count) loss
        
        #classification + poisson loss
        y_true_z = y_true[:,:,0]
        y_true_S = y_true[:,:,1]
        
        y_pred_z = y_pred[:,:,0]
        y_pred_S = y_pred[:,:,1]
        
        cls_loss = - y_true_S*tf.math.log(y_pred_S) - (1-y_true_S)*tf.math.log(1-y_pred_S)
        reg_loss = y_pred_z - y_true_z * tf.math.log(y_pred_z)
        loss = cls_loss + gamma * reg_loss * (1 - y_true_S)
        
        return(loss)
    
class VanillaLSTMStoreItem():
    def __init__(self, num_layers=None, num_units=None, input_window=None, output_window=None, encoder_exog_size=None, decoder_exog_size=None, lr=1e-3, dropout_rate=0, l2_regu=l2(1e-6), model=None, dataset_generator=None):
        self.num_layers = num_layers
        self.num_units = num_units
        self.input_window = input_window
        self.output_window = output_window
        self.encoder_exog_size = encoder_exog_size
        self.decoder_exog_size = decoder_exog_size
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.l2_regu = l2_regu
        
        self.model = model
        self.dataset_generator = dataset_generator #Provide the load method for training & inference of model
    
    def build_model(self):
        """
        Build Vanilla LSTM encoder-decoder network.
        """
        num_layers, num_units, input_window, output_window, encoder_exog_size, decoder_exog_size, dropout_rate, l2_regu =\
        self.num_layers, self.num_units, self.input_window, self.output_window, self.encoder_exog_size, self.decoder_exog_size, self.dropout_rate, self.l2_regu
        
        #Define embedding layers (item_id, event_name)
        item_embed = Embedding(input_dim=3049, output_dim=128, mask_zero=False, name='item_embed')
        event_embed = Embedding(input_dim=31, output_dim=8, mask_zero=False, name='event_embed')
        
        #Define encoder model
        encoder_input = Input(shape=(input_window, 1))
        encoder_exog_input = Input(shape=(input_window, encoder_exog_size))
        
        encoder_concat_input = Concatenate()([encoder_input, encoder_exog_input])
        
        encoder_lstm_res = {}
        for i in range(num_layers):
            encoder_lstm = LSTM(num_units[i], kernel_regularizer=l2_regu, recurrent_regularizer=l2_regu, dropout=dropout_rate, recurrent_dropout=0,
                                return_sequences=True, return_state=True, name='encoder_lstm_{}'.format(i))
            if (i == 0):
                encoder_lstm_outputs, encoder_lstm_state_h, encoder_lstm_state_c = encoder_lstm(encoder_concat_input)
            else:
                encoder_lstm_outputs, encoder_lstm_state_h, encoder_lstm_state_c = encoder_lstm(encoder_lstm_res[(i-1, 'outputs')])

            encoder_lstm_res[(i, 'model')] = encoder_lstm
            encoder_lstm_res[(i, 'outputs')] = encoder_lstm_outputs
            encoder_lstm_res[(i, 'states')] = [encoder_lstm_state_h, encoder_lstm_state_c]

        #Define decoder model
        #endog input for decoder. It is always a vector of 0s, meaning that model is trained unconditionally without using any forecast information.
        decoder_input = Input(shape=(output_window, 1))
        decoder_exog_input = Input(shape=(output_window, decoder_exog_size))
        
        decoder_item_input = Input(shape=(output_window,))
        decoder_item_embed = item_embed(decoder_item_input)
        
        decoder_event_input = Input(shape=(output_window,))
        decoder_event_embed = event_embed(decoder_event_input)
        
        decoder_concat_input = Concatenate()([decoder_input, decoder_exog_input, decoder_item_embed, decoder_event_embed])
        
        decoder_lstm_res = {}
        for i in range(num_layers):
            decoder_lstm = LSTM(num_units[i], kernel_regularizer=l2_regu, recurrent_regularizer=l2_regu, dropout=dropout_rate, recurrent_dropout=0,
                                return_sequences=True, return_state=True, name='decoder_lstm_{}'.format(i))
            if (i == 0):
                decoder_lstm_outputs, _, _ = decoder_lstm(decoder_concat_input, initial_state=encoder_lstm_res[(i, 'states')])
            else:
                decoder_lstm_outputs, _, _ = decoder_lstm(decoder_lstm_res[(i-1, 'outputs')], initial_state=encoder_lstm_res[(i, 'states')])

            decoder_lstm_res[(i, 'model')] = decoder_lstm
            decoder_lstm_res[(i, 'outputs')] = decoder_lstm_outputs
        
        decoder_dense_z = Dense(16, activation='tanh', kernel_regularizer=l2_regu, name='decoder_dense_z')(decoder_lstm_outputs)
        decoder_output_z = Dense(1, activation='exponential', kernel_regularizer=l2_regu, name='decoder_output_z')(decoder_dense_z) #For the count distribution
        
        decoder_dense_S = Dense(16, activation='tanh', kernel_regularizer=l2_regu, name='decoder_dense_S')(decoder_lstm_outputs)
        decoder_output_S = Dense(1, activation='sigmoid', kernel_regularizer=l2_regu, name='decoder_output_S')(decoder_dense_S) #For the is_zero distribution
        
        decoder_output = Concatenate()([decoder_output_z, decoder_output_S])

        #training mode of model
        model = Model(inputs = [encoder_input, encoder_exog_input, decoder_input, decoder_exog_input, decoder_item_input, decoder_event_input], outputs = decoder_output)
        adam = Adam(learning_rate=self.lr)
        model.compile(optimizer=adam, loss=loss_function)
        print(model.summary())
        
        self.model = model
        
        return(model)
    
    def prepare_training_dataset(self, train_endog, w, batch_size=256, sliding_freq=1):
        """
        Generator for constructing the training dataset from train_endog.
        """
        input_window, output_window = self.input_window, self.output_window

        timestamp = np.arange(input_window, train_endog.shape[0]-output_window+1, sliding_freq)
        while (1):
            permutated_timestamp = np.random.choice(timestamp, size=1, replace=False) #Random sample of a timestamp
            for t in permutated_timestamp:
                endog_slide = train_endog.iloc[(t-input_window):(t+output_window), :].T.dropna().copy()
                
                sample_batch_index = np.random.choice(endog_slide.shape[0], size=batch_size, replace=False) #Random sample of a batch of ts
                X_endog = endog_slide.iloc[sample_batch_index, :input_window]
                Y_endog = endog_slide.iloc[sample_batch_index, input_window:]

                X_decoder = np.zeros((*Y_endog.shape, 1)) #decoder endog input, which is always 0 as training unconditionally

                encoder_exog = self.dataset_generator.load_encoder_exog(X_endog.columns, X_endog.index)
                decoder_exog = self.dataset_generator.load_decoder_exog(Y_endog.columns, Y_endog.index)
                decoder_item = self.dataset_generator.load_item_id(Y_endog.columns, Y_endog.index)
                decoder_event = self.dataset_generator.load_event_name(Y_endog.columns, Y_endog.index)
                
                Y_endog_S = np.where(Y_endog==0, 1, 0).reshape(*Y_endog.shape, 1)
                
                yield([[np.array(X_endog).reshape((*X_endog.shape,1)), encoder_exog, X_decoder, decoder_exog, decoder_item, decoder_event], 
                       np.concatenate([np.array(Y_endog).reshape(*Y_endog.shape, 1), Y_endog_S], axis=-1),
                       np.array(w[X_endog.index])])
    
    def get_forecast(self, input_endog, kind='expectation'):
        """
        Return forecast using input_endog.
        If kind = 'expectation': Return the expected value of forecast
        If kind = 'separate': Return a list of 2 forecast_df = [forecast_df_z, forecast_df_S]
        """
        output_window = self.output_window
        
        val_index = np.arange(input_endog.index[-1]+1, input_endog.index[-1]+1+output_window)

        X_endog = input_endog.T.dropna().copy()
        obs_items = X_endog.index #store_item pair that has complete input
        
        encoder_exog = self.dataset_generator.load_encoder_exog(X_endog.columns, obs_items)
        decoder_exog = self.dataset_generator.load_decoder_exog(val_index, obs_items)
        decoder_item = self.dataset_generator.load_item_id(val_index, obs_items)
        decoder_event = self.dataset_generator.load_event_name(val_index, obs_items)
        
        X_decoder = np.zeros((X_endog.shape[0], output_window, 1))

        forecast_zS = self.model.predict([np.array(X_endog).reshape((*X_endog.shape,1)), encoder_exog, X_decoder, decoder_exog, decoder_item, decoder_event],
                                          batch_size=256)
        
        if kind == 'expectation':
            forecast = forecast_zS[:,:,0] * (1 - forecast_zS[:,:,1]) #Expected value
            forecast_df = pd.DataFrame(np.NaN, index=val_index, columns=input_endog.columns)
            forecast_df[obs_items] = forecast.T
            return(forecast_df)
        
        elif kind == 'separate':
            forecast_z, forecast_S = forecast_zS[:,:,0], forecast_zS[:,:,1]
            
            forecast_df_z = pd.DataFrame(np.NaN, index=val_index, columns=input_endog.columns)
            forecast_df_z[obs_items] = forecast_z.T
            
            forecast_df_S = pd.DataFrame(np.NaN, index=val_index, columns=input_endog.columns)
            forecast_df_S[obs_items] = forecast_S.T
            return([forecast_df_z, forecast_df_S])
        
        
