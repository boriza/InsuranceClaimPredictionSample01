#%% [markdown]
# Now we will import pandas to read our data from a CSV file and manipulate it for further use. We will also use numpy to convert out data into a format suitable to feed our classification model. We'll use seaborn and matplotlib for visualizations. We will then import Logistic Regression algorithm from sklearn. This algorithm will help us build our classification model. Lastly, we will use joblib available in sklearn to save our model for future use.

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
#%% [markdown]
# Setup AML my reading AML configuration 
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core import ScriptRunConfig

run = Run.get_context()

#%% [markdown]
#make a log entiry to the AML experiement 
run.log(name="message", value="Hello world log!")

#%% [markdown]
# We have our data saved in a CSV file called insurance2.csv. We first read our dataset in a pandas dataframe called insuranceDF, and then use the head() function to show the first five records from our dataset.

#%%
insuranceDF = pd.read_csv('insurance2.csv')
print(insuranceDF.head())

#%% [markdown]
# The following features have been provided to help us predict whether a person is diabetic or not:
# 
# age : age of policyholder
# sex: gender of policy holder (female=0, male=1)
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 25
# children: number of children / dependents of policyholder
# smoker: smoking state of policyholder (non-smoke=0;smoker=1) 
# region: the residential area of policyholder in the US (northeast=0, northwest=1, southeast=2, southwest=3)
# charges: individual medical costs billed by health insurance
#%% [markdown]
# Let's also make sure that our data is clean (has no null values, etc).

#%%
insuranceDF.info()

#%% [markdown]
# Let's start by finding correlation of every pair of features (and the outcome variable), and visualize the correlations using a heatmap.

#%%
corr = insuranceDF.corr()
print(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plt.savefig('heatmap-01.png')
run.log_image("Heatmap", path ="heatmap-01.png")

#%% [markdown]
# In the above heatmap, brighter colors indicate more correlation.
#%% [markdown]
# When using machine learning algorithms we should always split our data into a training set and test set. (If the number of experiments we are running is large, then we can should be dividing our data into 3 parts, namely - training set, development set and test set). In our case, we will also separate out some data for manual cross checking.
# 
# The data set consists of record of 1338 patients in total. To train our model we will be using 1000 records. We will be using 300 records for testing, and the last 38 records to cross check our model.

#%%
dfTrain = insuranceDF[:1000]
dfTest = insuranceDF[1000:1300]
dfCheck = insuranceDF[1300:] 

#%% [markdown]
# Next, we separate the label and features (for both training and test dataset). In addition to that, we will also convert them into NumPy arrays as our machine learning algorithm process data in NumPy array format.

#%%
trainLabel = np.asarray(dfTrain['insuranceclaim'])
trainData = np.asarray(dfTrain.drop('insuranceclaim',1))
testLabel = np.asarray(dfTest['insuranceclaim'])
testData = np.asarray(dfTest.drop('insuranceclaim',1))

#%% [markdown]
# As the final step before using machine learning, we will normalize our inputs. Machine Learning models often benefit substantially from input normalization. It also makes it easier for us to understand the importance of each feature later, when we'll be looking at the model weights. We'll normalize the data such that each variable has 0 mean and standard deviation of 1.

#%%
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
 
trainData = (trainData - means)/stds
testData = (testData - means)/stds

#%% [markdown]
# We can now train our classification model. We'll be using a machine simple learning model called logistic regression. Since the model is readily available in sklearn, the training process is quite easy and we can do it in few lines of code. First, we create an instance called insuranceCheck and then use the fit function to train the model.

#%%
insuranceCheck = LogisticRegression()
insuranceCheck.fit(trainData, trainLabel)

#%% [markdown]
# Now use our test data to find out accuracy of the model.

#%%
accuracy = insuranceCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")

#%% [markdown]
# To get a better sense of what is going on inside the logistic regression model, we can visualize how our model uses the different features and which features have greater effect.

#%%
coeff = list(insuranceCheck.coef_[0])
labels = list(dfTrain.drop('insuranceclaim',1).columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

run.log_image("Importance", plot = plt)	

#%% [markdown]
# From the above figure, we can draw the following conclusions.
# 
# 1. BMI, Smoker have significant influence on the model, specially BMI. It is good to see our machine learning model match what we have been hearing from doctors our entire lives!
# 
# 2. Children has a negative influence on the prediction, i.e. higher number children / dependents are correlated with a policyholder not taken insurance claim.
# 
# 3. Although age was more correlated than BMI to the output variables (as we saw during data exploration), the model relies more on BMI. This can happen for several reasons, including the fact that the correlation captured by age is also captured by some other variable, whereas the information captured by BMI is not captured by other variables.
# 
# Note that this above interpretations require that our input data is normalized. Without that, we can't claim that importance is proportional to weights.
#%% [markdown]
# Now save our trained model for future use using joblib.

#%%
joblib.dump([insuranceCheck, means, stds], 'insurance01Model.pkl')

#%% [markdown]
# To check whether we have saved the model properly or not, we will use our test data to check the accuracy of our saved model (we should observe no change in accuracy if we have saved it properly).

#%%
insuranceLoadedModel, means, stds = joblib.load('insurance01Model.pkl')
accuracyModel = insuranceLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")

#%% [markdown]
# Now use our unused 38 data to see how predictions can be made. We have our unused data in dfCheck.

#%%
print(dfCheck.head(38))

#%% [markdown]
# Now use the third record to make our insurance claim prediction.

#%%
sampleData = dfCheck[2:3]
 
# prepare sample  
sampleDataFeatures = np.asarray(sampleData.drop('insuranceclaim',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
 
# predict 
predictionProbability = insuranceLoadedModel.predict_proba(sampleDataFeatures)
prediction = insuranceLoadedModel.predict(sampleDataFeatures)
print('Insurance Claim Probability:', predictionProbability)
print('Insurance Claim Prediction:', prediction)

run.complete()
print(run.get_status())
