# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest Classifier used to predict if an individuals salary is over 
or under 50K based on census data. The model was trained on approximately 30,
000 rows of data which is all publicly available. The data has been cleaned 
by hand (primarily removing extenuous spaces and missing data).

## Intended Use
This model is served up using a REST API to allow users to provide their own 
data and it will predict their salary range (>50K, <=50K). The documentation 
for the rest api is available at https://sam-course-ml-app.herokuapp.
com/docs. 

## Training Data
The training data was taken from openly available US Census data, provided by 
the udacity course website.  ~80% of the 29,969 rows  of data were used 
for training with the remaining 20% used for testing the model.

Data Headers:
'age,workclass,fnlgt,education,education-num,marital-status,occupation,
relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,
salary
'
An example row of data:
'52,Self-emp-inc,287927,HS-grad,9,Married-civ-spouse,Exec-managerial,Wife,
White,Female,15024,0,40,United-States,>50K'


## Evaluation Data
Hold-Out approach where 20% of the data has been split off by means to 
evaluate the models accuracy. This could be improves using K-Fold.

## Metrics
Precision 0.7311669128508124 
recall 0.6595602931379081 
fbeta 0.6935201401050789


## Ethical Considerations
Data set contains data which is often used to discriminate such as age, 
race and gender.  It is important the data and model are both evaluated for 
bias on all of these fields

US Census data is also strongly affected by bias, as workers such as 
undocumented immigrants and native tribes are considered unrepresentative.

## Caveats and Recommendations
If this model were to be taken further I would recommend turning it into a 
pipeline so that it could be retrained as further census information 
becomes available.

As mentioned previously, the model would benefit from using K-fold instead 
of hold out. Further more it has not undergone hyperparameter tuning which 
could see a significant improvement in the accuracy.