# Module-21-NN-Deep-Learning-

# Overview

In this assignment, we utilised a Neural Network model to predict applicants with the best chance of success for funding. The applicants were organisations that applied to the non-profit foundation Alphabet Soup for funding. The data set was a CSV file with over 34,000 organisations that have received funding over the year. The CSV file was read into a pandas dataframe from a [web source](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv).

The fields of the data source were as follows:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 34299 entries, 0 to 34298
Data columns (total 12 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   EIN                     34299 non-null  int64 
 1   NAME                    34299 non-null  object
 2   APPLICATION_TYPE        34299 non-null  object
 3   AFFILIATION             34299 non-null  object
 4   CLASSIFICATION          34299 non-null  object
 5   USE_CASE                34299 non-null  object
 6   ORGANIZATION            34299 non-null  object
 7   STATUS                  34299 non-null  int64 
 8   INCOME_AMT              34299 non-null  object
 9   SPECIAL_CONSIDERATIONS  34299 non-null  object
 10  ASK_AMT                 34299 non-null  int64 
 11  IS_SUCCESSFUL           34299 non-null  int64 
dtypes: int64(4), object(8)
memory usage: 3.1+ MB
```
The EIN and NAME columns were dropped as they were a unique ID and the name of the organisation respectively. There were 7 other categorical data columns and 3 numeric data columns. All the categorical columns had less than 10 distinct values except the `APPLICATION_TYPE` (17 Distinct Values) and `CLASSIFICATION` (71 Distinct Values) columns. These two columns were transformed to reduce the number of distinct variables by grouping them as 'other' and reducing the total number of distinct variables in each column.

The label we were trying to predict was the `IS_SUCCESSFUL` column (y value), and the rest of the dataframe (after removing the `EIN` and `NAME` columns) formed the features (x value). The categorical variables were converted to numeric using the `get_dummies` function in Pandas. The features were then scaled using the 'StandardScaler' from SciKit Learn Library.

We built a model using the TensorFlow module. The dataset was split using the Scikit Learn library for training and testing the model. We ran the model three times, and in each run, we varied either the data or the model slightly to optimise for accuracy. We also ran the Keras Tuner to find an optimum-performing model. The output model files for each of the attempts are saved in the [output folder](output).

# Results

## Attempt 1
This is the original attempt, and the code for this can be found in the Jupyter Notebook called [AphabetSoupCharity.ipynb](AphabetSoupCharity.ipynb)

**Data**: We reduced the distinct variables in the `APPLICATION_TYPE` from 17 to 10. We also reduced the `CLASSIFICATION` distinct variables from 71 to 9

**Model**: The model had a first layer with 6 nodes and a second hidden layer with 4 nodes. We used the 'relu' activation and trained with a 100 epochs

We achieved the following results from the model:
```
Loss: 0.554628312587738, Accuracy: 0.7269970774650574
```
The model had an accuracy of 72.7%.

## Attempt 2

To improve on the original attempt we adjusted the model to optimise the model for better accuracy. This time we increased the number of nodes in the layers. The code for this second attempt can be found in [AphabetSoupCharity_Optimisation_Attempt_1.ipynb](AphabetSoupCharity_Optimisation_Attempt_1.ipynb)

**Data**: The data was left the same as for Attempt 1.

**Model**: The model had a first layer with 10 nodes and a second hidden layer with 10 nodes. We used the 'relu' activation and trained again with 100 epochs.

We achieved the following results from the model:
```
Loss: 0.5522013902664185, Accuracy: 0.7300291657447815
```
The model's accuracy was 73%, which is more or less the same as the first attempt. The increase in the number of nodes in the first and second layers did not impact the accuracy.

## Attempt 3

Further changes to the data and the model were undertaken to see if accuracy could be improved.

**Data**:  In the `APPLICATION_TYPE` column the distinct variables were further reduced from 17 to 10 to 9. We also reduced the `CLASSIFICATION` distinct variables further from 71 to 9 to 6

**Model**: The model had a first layer with 80 nodes, a second hidden layer with 50 nodes and a third hidden layer with 50 nodes. We used the 'relu' activation and trained again with 100 epochs.

We achieved the following results from the model:
```
Loss: 0.5786693096160889, Accuracy: 0.7294460535049438
```

Interestingly, we had a slightly reduced accuracy (72.9%) despite increasing the layers and nodes, and we suspect this is probably because of the change in the data grouping.

## Keras Tuner

The assignment task was to optimise the model and see if we could achieve over 75% Accuracy. The code for this can be found in [AphabetSoupCharity_Optimisation_KerasTuner.ipynb](AphabetSoupCharity_Optimisation_KerasTuner.ipynb). We attempted to use the Keras Tuner to find the optimal performing model. The best model is shown below:
```
{'activation': 'relu',
 'first_units': 21,
 'num_layers': 5,
 'units_0': 51,
 'units_1': 51,
 'units_2': 41,
 'units_3': 41,
 'units_4': 71,
 'units_5': 31,
 'tuner/epochs': 20,
 'tuner/initial_epoch': 7,
 'tuner/bracket': 2,
 'tuner/round': 2,
 'tuner/trial_id': '0044'}
```

The best model consisted of 1st layer with 21 nodes and 5 hidden layers with the following number of nodes 51, 51, 41, 41 and 71. The output layer had 31 nodes. The activation used for this model was 'Relu'. Other models that were compared to the 'Relu' were 'tanh' and 'sigmoid'. We kept the epoch to 20 to reduce the computing load on the desktop PC (Note: The code was locally run on a PC and not committed to Google CoLab).

This model had the following results:
```
Loss: 0.5564839243888855, Accuracy: 0.7343440055847168
```
This is by far the best result we have had in all the attempts with an Accuracy of 73.4%

# Summary

Given all the attempts at predicting the 'successful' applicant, the best model we had managed to build only had an accuracy score of 73.4%.  The assignment is looking for a binary outcome, and perhaps other machine learning modes such as Logistic Regression, Random Forests, K-Nearest Neighbours, and Support Vector Machines could be trialled to see if they are any better at predicting. We could also potentially look at improving the input dataset in both quantity and quality. We could ask for a few extra years' worth of data, but the quality of data may be more crucial than just quantity. If we are able to better understand how Alphabet Soup Foundation is defining success, we could potentially seek out parameters which feed into that definition of success. Feeding these extra parameters as columns to the dataset may help the model to predict better predictions. 
