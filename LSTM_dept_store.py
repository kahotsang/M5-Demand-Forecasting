"""
Class implementation for training LSTM on dept_store level
There are two types of data: endog (the historical time-series / lag-features) and exog (features other than lag-features)
exog data is prepared through DatasetGeneratorDeptStore class
endog data is prepared through agg_endog (on dept_store level)

model is trained using VanillaLSTMDeptStore class, 
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

class DatasetGeneratorDeptStore():
    """
    Provide exog dataset generation method (load_) for training & inference of VanillaLSTMDeptStore
    """
    def __init__(self, calendar, sales_train, endog, prices_df):
        self.calendar = calendar.copy()
        self.sales_train = sales_train.copy()
        self.endog = endog.copy()
        self.prices_df = prices_df.copy()
        
        self.construct_agg_endog()
        self.prepare_dataset_encoder()
    
    def construct_agg_endog(self):
        """
        Construct agg_endog on dept_store level
        """
        agg_endog, ratio_df = construct_grouped_ts(self.sales_train, self.endog, agg_1='dept_id', agg_2='store_id', drop_inactive=True, return_ratio=True)
        agg_sales_train = self.sales_train.groupby(['dept_id', 'store_id']).size().reset_index() #indicator for dept_store_id
        agg_idx = ratio_df.agg_idx
        
        self.agg_endog = agg_endog.copy()
        self.ratio_df = ratio_df.copy()
        self.agg_sales_train = agg_sales_train.copy()
        self.agg_idx = agg_idx.copy()
        
    def prepare_dataset_encoder(self):
        """
        Prepare dataset and encoder for load methods
        """
        calendar, sales_train, prices_df = self.calendar, self.sales_train, self.prices_df
        agg_endog, agg_idx, agg_sales_train = self.agg_endog, self.agg_idx, self.agg_sales_train
        
        #Prepare exog dataset ---------------------------------------------------------------
        #Prepare calendar exog: event_type & wday on a date
        calendar_exog = pd.DataFrame(index=calendar.index)
        for event_type in ['Sporting', 'Cultural', 'National', 'Religious']:
            calendar_exog['is_{}'.format(event_type)] = np.where((calendar.loc[calendar_exog.index, ['event_type_1', 'event_type_2']] == event_type).any(axis=1), 1, 0)
        wday_encoder = OneHotEncoder(drop='first', sparse=False) #drop Sat.
        wday_df = pd.DataFrame(wday_encoder.fit_transform(calendar.loc[calendar_exog.index, ['wday']]), columns=['w7'] + ['w{}'.format(i) for i in range(1,6)])
        calendar_exog = pd.concat([calendar_exog, wday_df], axis=1)
        
        #Prepare snap_exog: if there is snap event on that date & dept_store ts
        snap_exog = pd.DataFrame(0., index=calendar.index, columns=agg_endog.columns)
        for idx in snap_exog.columns:
            state = sales_train[agg_idx == idx].state_id.unique()[0]
            snap_exog[idx] = calendar.loc[snap_exog.index, 'snap_{}'.format(state)]
        
        #Prepare price discount on that date & dept_store ts
        price_exog = pd.DataFrame(index=calendar.index, columns=agg_endog.columns) #mean price across item_store for a dept_store ts
        for idx in price_exog.columns:
            price_exog[idx] = prices_df.T.loc[agg_idx == idx].mean()
        price_discount = price_exog / price_exog.max() #normalized
        
        self.calendar_exog = calendar_exog
        self.snap_exog = snap_exog
        self.price_discount = price_discount
        
        #Prepare encoder ----------------------------------------------------------------------
        #Create encoder for dept_store_id
        dept_store_encoder = OneHotEncoder(drop='first', sparse=False).fit(agg_sales_train[['dept_id', 'store_id']])
        
        #Create encoder for event name
        calendar['event_name_1'].fillna('missing', inplace=True)
        event_encoder = LabelEncoder().fit(calendar['event_name_1'])
        
        self.dept_store_encoder = dept_store_encoder
        self.event_encoder = event_encoder
    
    def load_encoder_exog(self, exog_index, exog_cols):
        """
        Load encoder_exog at timestamp (exog_index), dept_stores (exog_columns).
        encoder_exog with shape (n_ts, n_timestamp, n_features)
        
        Currently, no encoder exog is used for dept_store_level, thus is set all 0.
        """
        encoder_exog = np.zeros((exog_cols.shape[0], exog_index.shape[0], 1))
        return(encoder_exog.astype(np.float32))
    
    def load_decoder_exog(self, exog_index, exog_cols):
        """
        Load decoder_exog at timestamp (exog_index), dept_stores (exog_columns).
        decoder_exog with shape (n_ts, n_timestamp, n_features)
        """
        decoder_exog = np.repeat(np.array(self.calendar_exog.loc[exog_index]).reshape((1,exog_index.shape[0],-1)), repeats=exog_cols.shape[0], axis=0)
        decoder_exog = np.concatenate([decoder_exog, np.array(self.snap_exog.loc[exog_index, exog_cols].T).reshape((exog_cols.shape[0], -1, 1))], axis=-1)
        decoder_exog = np.concatenate([decoder_exog, np.array(self.price_discount.loc[exog_index, exog_cols].T).reshape((exog_cols.shape[0], -1, 1))], axis=-1)
        
        #Append with id features
        id_features =  self.dept_store_encoder.transform(self.agg_sales_train.loc[exog_cols, ['dept_id', 'store_id']])
        id_features = np.repeat(np.expand_dims(id_features, axis=1), repeats=exog_index.shape[0], axis=1)
        decoder_exog = np.concatenate([decoder_exog, id_features], axis=-1)
        
        return(decoder_exog.astype(np.float32))
    
    def load_event_name(self, exog_index, exog_cols):
        """
        Load event_id at timestamp (exog_index), dept_stores (exog_columns).
        event_name with shape (n_ts, n_timestamp).
        """
        
        event_name = self.event_encoder.transform(self.calendar.loc[exog_index, 'event_name_1']).reshape((1,-1))
        event_name = np.repeat(event_name, repeats=exog_cols.shape[0], axis=0)
        
        return(event_name)

   
class VanillaLSTMDeptStore():
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
        
        #Define embedding layers (item_id, event_name), in case the embedding layers are applied to both encoder and decoder.
        event_embed = Embedding(input_dim=31, output_dim=8, mask_zero=False, name='event_embed')
        
        #Define encoder model
        encoder_input = Input(shape=(input_window, 1)) #endog input for encoder
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
        
        decoder_event_input = Input(shape=(output_window,))
        decoder_event_embed = event_embed(decoder_event_input)
        
        decoder_concat_input = Concatenate()([decoder_input, decoder_exog_input, decoder_event_embed])
        
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

        decoder_output = Dense(1, activation=None, kernel_regularizer=l2_regu, name='decoder_output')(decoder_lstm_outputs)

        #training mode of model
        model = Model(inputs = [encoder_input, encoder_exog_input, decoder_input, decoder_exog_input, decoder_event_input], outputs = decoder_output)
        adam = Adam(learning_rate=self.lr)
        model.compile(optimizer=adam, loss='mse')
        print(model.summary())
        
        self.model = model
        
        return(model)
    
    def prepare_training_dataset(self, train_endog, w, sliding_freq=1):
        """
        Prepare full dataset for training
        """

        input_window, output_window = self.input_window, self.output_window
        timestamp = np.arange(input_window, train_endog.shape[0]-output_window+1, sliding_freq)

        X_endog_f, Y_endog_f, X_decoder_f, encoder_exog_f, decoder_exog_f, decoder_event_f, w_f = [], [], [], [], [], [], []
        for t in timestamp:
            endog_slide = train_endog.iloc[(t-input_window):(t+output_window), :].T.dropna().copy()
            endog_slide = endog_slide - np.array(endog_slide.iloc[:,0]).reshape((-1,1)) #remove the first obs as to remove trend
            sample_batch_index = np.arange(endog_slide.shape[0])

            X_endog = endog_slide.iloc[sample_batch_index, :input_window]
            Y_endog = endog_slide.iloc[sample_batch_index, input_window:]

            X_decoder = np.zeros((*Y_endog.shape, 1)) #decoder endog input, which is always 0 as training unconditionally

            encoder_exog = self.dataset_generator.load_encoder_exog(X_endog.columns, X_endog.index)
            decoder_exog = self.dataset_generator.load_decoder_exog(Y_endog.columns, Y_endog.index)
            decoder_event = self.dataset_generator.load_event_name(Y_endog.columns, Y_endog.index)

            X_endog_f.append(np.array(X_endog).reshape((-1, input_window, 1)))
            Y_endog_f.append(np.array(Y_endog).reshape((-1, output_window, 1)))
            X_decoder_f.append(X_decoder)
            encoder_exog_f.append(encoder_exog)
            decoder_exog_f.append(decoder_exog)
            decoder_event_f.append(decoder_event)
            w_f.append(np.array(w[X_endog.index]))
                
        X_endog_f = np.concatenate(X_endog_f, axis=0)
        Y_endog_f = np.concatenate(Y_endog_f, axis=0)
        X_decoder_f = np.concatenate(X_decoder_f, axis=0)
        encoder_exog_f = np.concatenate(encoder_exog_f, axis=0)
        decoder_exog_f = np.concatenate(decoder_exog_f, axis=0)
        decoder_event_f = np.concatenate(decoder_event_f, axis=0)
        w_f = np.concatenate(w_f, axis=0)

        return([[X_endog_f, encoder_exog_f, X_decoder_f, decoder_exog_f, decoder_event_f], Y_endog_f, w_f])
    
    def get_forecast(self, input_endog):
        """
        Return forecast using input_endog.
        """
        output_window = self.output_window
        
        val_index = np.arange(input_endog.index[-1]+1, input_endog.index[-1]+1+output_window)

        X_endog = input_endog.T.dropna().copy()
        obs_items = X_endog.index #store_item pair that has complete input
        first_obs = np.array(X_endog.iloc[:,0]).reshape((-1,1)) #remove the first obs to remove trend
        X_endog -= first_obs
        
        encoder_exog = self.dataset_generator.load_encoder_exog(X_endog.columns, obs_items)
        decoder_exog = self.dataset_generator.load_decoder_exog(val_index, obs_items)
        decoder_event = self.dataset_generator.load_event_name(val_index, obs_items)

        X_decoder = np.zeros((X_endog.shape[0], output_window, 1))

        forecast = self.model.predict([np.array(X_endog).reshape((*X_endog.shape,1)), encoder_exog, X_decoder, decoder_exog, decoder_event]).reshape((-1, output_window))
        
        #Add back the first obs
        forecast += first_obs
        forecast_df = pd.DataFrame(np.NaN, index=val_index, columns=input_endog.columns)
        forecast_df[obs_items] = forecast.T
        return(forecast_df)