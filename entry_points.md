#MODEL BUILD:

1. re-train model & re-compute model residuals estimate, then predict using re-trained model
    
    i. expect to run for 8 - 12 hrs
    
    ii. train all models from scratch
    
    iii. forecast using re-trained model
    
    iv. able to forecast on new forecast period (using new training dataset, the forecast period is still the 28D since the last day of training data)

Steps to run the build is as follows:

#1) Run the following scripts after data setup:

python ./prepare_data.py

python ./train_store_item.py

python ./train_dept_store.py

python ./predict.py
