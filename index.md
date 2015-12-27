# Prediction Assignment Writeup
Stefan  
`r format(Sys.time(), '%Y-%m-%d')`  

###Background
This is the writeup for the Coursera course Practical Machine Learning.
The task of this writeup is to predict the manner of how weight lifting exercises is performed in the data described in [this study]( http://groupware.les.inf.puc-rio.br/har)  

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
##          A 1383   14    1    0    0
##          B    7  920   19    0    4
##          C    5   14  822   24    9
##          D    0    0   12  779    4
##          E    0    1    1    1  884
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9763          
##                  95% CI : (0.9717, 0.9804)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9701          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9914   0.9694   0.9614   0.9689   0.9811
## Specificity            0.9957   0.9924   0.9872   0.9961   0.9993
## Pos Pred Value         0.9893   0.9684   0.9405   0.9799   0.9966
## Neg Pred Value         0.9966   0.9927   0.9918   0.9939   0.9958
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2820   0.1876   0.1676   0.1588   0.1803
## Detection Prevalence   0.2851   0.1937   0.1782   0.1621   0.1809
## Balanced Accuracy      0.9936   0.9809   0.9743   0.9825   0.9902
```
The accuracy of the model is 97.63%.  
95 percent conficence interval of accuracy is 97.17% to 
98.04%.  

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

