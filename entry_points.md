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

