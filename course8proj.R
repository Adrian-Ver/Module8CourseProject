list.files()
library(caret)
library(dplyr)

pml_training <- read.csv("pml-training.csv")
pml_testing <- read.csv("pml-testing.csv")
pml_training$classe <- as.factor(pml_training$classe)

# descriptives for the training data set
dim(pml_training); dim(pml_testing)
names(pml_training)
table(pml_training$classe)
round(100*prop.table(
    table(pml_training$user_name, pml_training$classe),
    margin = 1), digits = 4)

# DATA CLEANING
#  removing irrelevant columns (timestamp and NA columns)
pml_training <- pml_training[,colSums(is.na(pml_training)) == 0]
pml_testing <- pml_testing[,colSums(is.na(pml_training)) == 0]
pml_training <- pml_training %>% 
    select(-contains("timestamp")) %>%
    select(-contains("skewness")) %>%
    select(-contains("kurtosis")) %>%
    select(-contains("max")) %>%
    select(-contains("min")) %>%
    select(-contains("amplitude"))
pml_testing <- pml_testing %>% 
    select(-contains("timestamp")) %>%
    select(-contains("skewness")) %>%
    select(-contains("kurtosis")) %>%
    select(-contains("max")) %>%
    select(-contains("min")) %>%
    select(-contains("amplitude"))

# creating training and testing data sets within training data set
set.seed(1975)
inTrain <- createDataPartition(pml_training$classe, p = 0.7, list = F)
pml_training_train <- pml_training[inTrain,]
pml_training_test <- pml_training[-inTrain,]

# method 1: creating a simple random tree
set.seed(1975)
pml_fit_rpart <- train(classe ~ . , data = pml_training_train, method = "rpart")
pml_pred_rpart <- predict(pml_fit_rpart, newdata = pml_training_test)
(cm_rpart <- confusionMatrix(pml_pred_rpart, pml_training_test$classe))

# method 2: creating a random forest
set.seed(1975)
pml_fit_rf <- train(classe ~ . , data = pml_training_train, method = "rf")
pml_pred_rf <- predict(pml_fit_rf, newdata = pml_training_test)
(cm_rf <- confusionMatrix(pml_pred_rf, pml_training_test$classe))

# method 3: creating repeated cross-validation within training set
set.seed(1975)
pml_fit_cv <- train(classe ~ . , data = pml_training_train, method = "rpart", trControl = trainControl(method = "cv", number = 5))
pml_pred_cv <- predict(pml_fit_cv, newdata = pml_training_test)
(cm_cv <- confusionMatrix(pml_pred_cv, pml_training_test$classe))

# method 4: using gradient boosting method
set.seed(1975)
pml_fit_gbm <- train(classe ~ . , data = pml_training_train, method = "gbm")
pml_pred_gbm <- predict(pml_fit_gbm, newdata = pml_training_test)
(cm_gbm <- confusionMatrix(pml_pred_gbm, pml_training_test$classe))

# predict on the test data set
pml_pred_test <- predict(pml_fit_rf, newdata = pml_testing)

