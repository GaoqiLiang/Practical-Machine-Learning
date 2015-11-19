# Practical Machine Learning Course Project
## Background
### Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
### This report should include: (1) How you build your model; (2) How you used cross validataion; (3) What you think the expected out of sample error is; (4) Why you made the choices you did.


## Preprocessing
# Download files 
library (httr)
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_raw_traning <- "raw_traning.csv"
if(!file.exists(url_raw_traning)){
  download.file(url, url_raw_traning)
}

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
url_raw_testing <- "raw_testing.csv"
if(!file.exists(url_raw_testing)){
  download.file(url, url_raw_testing)
}

# Read data
training <- read.csv(url_raw_traning, na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url_raw_testing, na.strings=c("NA","#DIV/0!",""))
dim(training)
dim(testing)

## Build Model
# Load the package
library(caret)
# Partioning the training set with 60% myTraining and 40% myTesting
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ] 
myTesting <- training[-inTrain, ]
dim(myTraining)
dim(myTesting)
head(myTraining)


# Get data clean
# Remove NearZeroVariance Variables (Total count 60 columns)
myDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)
myNZVvars <- names(myTraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
                                      "kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
                                      "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
                                      "var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
                                      "stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
                                      "kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
                                      "max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
                                      "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
                                      "skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
                                      "amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
                                      "skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
                                      "max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
                                      "amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
                                      "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
                                      "stddev_yaw_forearm", "var_yaw_forearm")
myTraining <- myTraining[!myNZVvars]
dim(myTraining)
# Remove the first ID variables
myTraining <- myTraining[c(-1)]
dim(myTraining)
head(myTraining)
# Remove the columns that have more than 60% NA
trainingV3 <- myTraining #creating another subset to iterate in loop
for(i in 1:length(myTraining)) { #for every column in the training dataset
  if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .6 ) { 
    for(j in 1:length(trainingV3)) {
      if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) ==1)  { #if the columns are the same:
        trainingV3 <- trainingV3[ , -j] #Remove that column
      }   
    } 
  }
}

myTraining <- trainingV3
dim(myTraining)


# Get data clean similarly for the myTesting data and testing data
clean1 <- colnames(myTraining)
myTesting <- myTesting[clean1]
dim(myTesting)

clean2 <- colnames(myTraining[, -58]) #already with classe column removed 
testing <- testing[clean2]
dim(testing)

head(myTesting)
head(testing)


# We need to make some ajustments to make the myTesting data and testing data have the same type
for (i in 1:length(testing) ) {
  for(j in 1:length(myTraining)) {
    if( length( grep(names(myTraining[i]), names(testing)[j]) ) ==1)  {
      class(testing[j]) <- class(myTraining[i])
    }      
  }      
}
# Make sure Coertion really worked, simple smart ass technique:
testing <- rbind(myTraining[2, -58] , testing) #note row 2 does not mean anything, this will be removed right.. now:
testing <- testing[-1,]
dim(testing)

## Decision Tree for ML algorithm using for prediction
library(rpart)
library(rpart.plot)
library(rattle)
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")
# Plot the decision tree:
fancyRpartPlot(modFitA1)

# Predicting:
predictionsA1 <- predict(modFitA1, myTesting, type = "class")
confusionMatrix(predictionsA1, myTesting$classe)
# we can see that in the confusion matrix and statistics, the accuracy is 87.79%

## Random Forests for ML algorithm using for prediction
library(randomForest)
modFitB1 <- randomForest(classe ~. , data=myTraining)
predictionsB1 <- predict(modFitB1, myTesting, type = "class")
confusionMatrix(predictionsB1, myTesting$classe)
# we can see that in the confusion matrix and statistics, the accuracy is 99.77%
# Therefore, by comparision, the random forest yielded better result accuracy. We will choose the random forest alogrithm to test the testing data.


## Apply the machine learning algorithm to the 20 test cases available in the test data 
predictionsB2 <- predict(modFitB1, testing, type = "class")
print(predictionsB2)













