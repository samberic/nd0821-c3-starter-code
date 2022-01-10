# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest Classifier used to predict if an individuals salary is over or under 50K based on census data

## Intended Use
This model will be used to pass the udacity course the author is currently taking

## Training Data
Census data, provided by the udacity course website

## Evaluation Data
Hold-Out approach where 20% of the data has been split off by means to evaluate the models accuracy

## Metrics
Precision 0.7311669128508124 
recall 0.6595602931379081 
fbeta 0.6935201401050789


## Ethical Considerations
Data set contains data which is often used to discriminate such as age, race and gender.  It is important the data and model are both evaluated for bias on all of these fields

## Caveats and Recommendations
If this model were to be taken further I would recommend turning it into a pipeline