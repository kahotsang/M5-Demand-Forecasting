# M5-Demand-Forecasting
Here you can find the outline to reproduce the my solution in M5 Walmart Demand Forecasting Competition. If you run into any trouble with the setup/code or have any questions please contact me at khtsangdavid@gmail.com

## HARDWARE
(The following specs were used to create the original solution)
- MacOS Catalina 10.15.5
- 1.4 GHz Quad-Core Intel Core i5 (4 Cores, 16GB memory)
- No GPU is required

## SOFTWARE
(python packages are detailed separately in 'requirements.txt')
- Python 3.7.6

## Data Setup
Download and unzip the data from M5 Forecasting - Uncertainty Competition. Place all the csv files into ./raw_data/

## Outline to make submission files
(All the script should be run on the top level directory)

#Data Preprocessing: (after data setup) The following script will preprocess data at ./raw_data/ and save the preprocessed data at ./processed_data/

python ./prepare_data.py

#Model Training: (after data preprocessing step) The following scripts will train two LSTM models (at store_item level & dept_store level) and save (and replace) the checkpoints & model residuals estimate at ./model/ and ./logs/ respectively

python ./train_store_item.py
python ./train_dept_store.py

#Prediction: The following script will forecast on the next 28D since the last day of the training data and save the submission.csv at ./submission/

python ./predict.py

#MODEL BUILD: There are two options to produce the solution.
1. prediction using pre-trained model
    a. run in several minutes
    b. uses pre-trained LSTM model (and the pre-computed model residuals estimate)
    c. only able to forecast the next 28D since the last day of training data (i.e. currently can only forecast from d_1942 to d_1969)
2. re-train model & re-compute model residuals estimate, then predict using re-trained model
    a. expect to run for 8 - 12 hrs
    b. train all models from scratch
    c. forecast using re-trained model
    d. able to forecast on new forecast period (using new training dataset, the forecast period is still the 28D since the last day of training data)

Steps to run each build is as follows:
#1) Run the following scripts after data setup:
python ./prepare_data.py
python ./predict.py

#2) Run the following scripts after data setup:
python ./prepare_data.py
python ./train_store_item.py
python ./train_dept_store.py
python ./predict.py
