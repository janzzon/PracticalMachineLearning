# Prediction Assignment Writeup
Stefan  
`r format(Sys.time(), '%Y-%m-%d')`  

###Background
The task of this writeup is to predict the manner of how weight lifting exercises is performed in the data described in [this study]( http://groupware.les.inf.puc-rio.br/har)  
This is the writeup for the Coursera course Practical Machine Learning.

###Loading the data


```r
# Prediction Assignment Writeup
# https://class.coursera.org/predmachlearn-035/human_grading/view/courses/975205/assessments/4/submissions
library(magrittr)
library(plyr)
library(dplyr)
library(downloader)
library(caret)
library(doParallel)
# create data dir and download files i missing
train_url <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
test_url <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
if(!file.exists("./data")) dir.create("./data")
if(!file.exists("data/pml-training.csv")) download(train_url, destfile = "data/pml-training.csv")
if(!file.exists("data/pml-testing.csv")) download(test_url, destfile = "data/pml-testing.csv")
# Read train and test files to data frames
train_rawdat <- read.csv("./data/pml-training.csv", stringsAsFactors = F, na.strings = c("","NA", "#DIV/0!"))
test_rawdat <- read.csv("./data/pml-testing.csv", stringsAsFactors = F, na.strings = c("","NA", "#DIV/0!"))
```

###Cleaning of data
According to the assignment instructions, the investigated variable to predict is "classe".  
Some of the columns in the training data set contains NA-values, these columns are removed.


```r
# Find column names with data containing NA-values
na_names <- names(train_rawdat)[lapply(train_rawdat, function(x) any(is.na(x))) %>% unlist]
# Column 1 to 7 doesn't contain useful data for prediction.
# Create vector with column 1:7 and columns with NA-data
exclude_names <- c(na_names,names(train_rawdat)[1:7])
# Create new data set without excluded columns
train_cleaned <- train_rawdat[,!(names(train_rawdat) %in% exclude_names)]
```

###Split training data
The training data is splitted into training and validation data

```r
# Make vector for creation of training and test data. Create training/test data sets
inTrain <- createDataPartition(y=train_cleaned$classe, p=0.75, list=FALSE)
training <- train_cleaned[inTrain,]
testing <- train_cleaned[-inTrain,]
```

###Model creation
The prediction model is Random Forest, pre processing is performed with principal component analysis.  
Random Forest is slow but tend to deliver good results with various data. This was the first model I tried, and it seems to be good enough.  
To speed up model training a bit, the training is utilizing multiple cores if available.  
The model is also saved as a .RDS-file. If there's an existing model file, that's used instead of recompute the model.  

```r
# Check if saved model file exists. Read if extists, train model and save .RDS file if missing.
if (file.exists("modFitPca.RDS")) {
  # Read model from file
  modFitPca <- readRDS("modFitPca.RDS")
} else {
  # Create model if model file is missing.
  # Set trainControl options
  tc <-
    trainControl(preProcOptions = list(thresh = 0.95), method = "cv")
  # Create cluster, (1 less than number of cores), but at least 1.
  cl <- makeCluster(max(1,detectCores() - 1))
  registerDoParallel(cl)
  # parallel code here
  modFitPca <-
    train(classe ~ ., method = "rf", trControl = tc , preProcess = "pca",
      data = training)
  # Stop cluster
  stopCluster(cl)
  # Save model to file
  saveRDS(modFitPca, "modFitPca.RDS")
}
```

```
## Loading required package: randomForest
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

###Validation and error rate

```r
# Predict new values from testing data set
prediction <- predict(modFitPca, testing)

# Look at test values vs. predicted values in test data
(cm_test_result <- confusionMatrix(prediction, testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1385   10    1    1    0
##          B    2  917   19    0    6
##          C    6   20  823   32    7
##          D    2    0   11  769    2
##          E    0    2    1    2  886
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9747          
##                  95% CI : (0.9699, 0.9789)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.968           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9928   0.9663   0.9626   0.9565   0.9834
## Specificity            0.9966   0.9932   0.9839   0.9963   0.9988
## Pos Pred Value         0.9914   0.9714   0.9268   0.9809   0.9944
## Neg Pred Value         0.9971   0.9919   0.9920   0.9915   0.9963
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2824   0.1870   0.1678   0.1568   0.1807
## Detection Prevalence   0.2849   0.1925   0.1811   0.1599   0.1817
## Balanced Accuracy      0.9947   0.9797   0.9733   0.9764   0.9911
```
The accuracy of the model is 97.47%.  
95 percent conficence interval of accuracy is 96.99% to 
97.89%.  

###Test cases
The submission files for the project is created with this code provided from Coursera.

```r
# Function to create answer files, from Coursera
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
# Apply model to original test dataset, test_rawdat
finalResult <- predict(modFitPca,test_rawdat)
pml_write_files(finalResult)
```

