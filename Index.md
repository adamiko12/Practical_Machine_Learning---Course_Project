#Practical Machine Learning Course Project

*Adam Hadar*  
*May 16, 2015*

##Backround

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
In this project, i will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.

##How the model was built

classe is a variable with 5 levels. "Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:     
Class A - exactly according to the specification  
Class B - throwing the elbows to the front  
Class C - lifting the dumbbell only halfway  
Class D - lowering the dumbbell only halfway  
Class E - throwing the hips to the front  

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg)." [1]

##Load Data

Loading the training and testing data from my working directory, making sure that missing values are coded correctly.

```{r results='hide', message=FALSE, warning=FALSE}
library(caret);library(randomForest);library(scales)
training_data <- read.csv("./pml-training.csv",  na.strings=c("NA","#DIV/0!",""))
test_data <- read.csv("./pml-testing.csv",  na.strings=c("NA","#DIV/0!",""))
```

```{r}
dim(training_data)
message("The training data set contains ",comma_format()(nrow(training_data)), " observations and ", ncol(training_data), " variables")
dim(test_data)
message("The testing data set contains ",(nrow(test_data)), " observations and ", ncol(test_data), " variables")
#The variable that will be predicted is classe, which is divided to five classes
summary(training_data$classe)
#Calculating the classe 19,622 observations ratio to see if it will be the same ratio as the final 20 observations ratio prediction
c <- percent(summary(training_data$classe)/sum(summary(training_data$classe)))
data.frame(Class_count = summary(training_data$classe), Ratio = c)
```


##Clean Data

Removing columns with NA missing values.

```{r}
training_data <- training_data[,colSums(is.na(training_data)) == 0]
test_data <- test_data[,colSums(is.na(test_data)) == 0]
```

##Set Data

The training set is divided in two parts, one for training (60% of the training data) and the other for cross validation (40% of the training data).

```{r}
#Set random seed for reproduceability
set.seed(1981)
intrain <- createDataPartition(y=training_data$classe, p=0.6, list=FALSE)
training <- training_data[intrain, ]
testing <- training_data[-intrain, ]
```

The first 7 columns in the data set will be removed because they're unnecessary for predicting.
Those columns are:  
- X  
- user_name  
- raw_timestamp_part_1  
- raw_timestamp_part_2  
- cvtd_timestamp  
- new_window  
- num_window  

```{r}
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
dim(training)
message("The new training data set contains ",comma_format()(nrow(training)), " observations and ", ncol(training), " variables")
dim(testing)
message("The new testing data set contains ",comma_format()(nrow(testing)), " observations and ", ncol(testing), " variables")
```

##Training

I decided to start with Random Forest algorithm, to see if it would have acceptable results. 

```{r}
model <- randomForest(classe ~. , data=training, method="class")
#Estimate the performance of the model on the validation data set
prediction <- predict(model, testing, type = "class")
#Show confusion matrix to get estimate of out-of-sample error
confusionMatrix(prediction, testing$classe)
```

The estimated accuracy is **99.49%** and the estimated out-of-sample error is **0.51%**, so it is safe to say that this Random Forest model fit well to the project.

##Submission

Applying the model to the original testing data set downloaded from the data source and predicting outcome levels using Random Forest algorithm.


```{r}
final_prediction <- predict(model, test_data, type="class")
final_prediction
```

```{r}
#Calculating the classe 20 observations ratio prediction to see if it close to the original ratio with the 19,622 observations
r <- percent(summary(final_prediction)/sum(summary(final_prediction)))
data.frame(class_count = summary(final_prediction),prediction_ratio = r)
#The two ratios are very different, but its logical beacuse 20 observations does not behave the same as 19,622 observations
```

##References
*[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13) . Stuttgart, Germany: ACM SIGCHI, 2013.*
