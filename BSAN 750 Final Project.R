##########################################
#Logicsitc model
heart<-read.csv("heart.csv")
##### Reading in data and creating training/test set
heart<-heart[,-1]
index<- sample(nrow(heart),nrow(heart)*0.75)
train<-heart[index,]
test<-heart[-index,]

install.packages("glmnet")
library(glmnet)

###### Variable selection using a logistic model 
y.ind <- which(names(train)=="target")
trainX <- scale(train[, -y.ind])
trainY <- train[, y.ind]
testX <- scale(test[, -y.ind])
testY <- test[, y.ind]
##### Lasso fit and cross validation
lasso.fit <- glmnet(x=as.matrix(trainX), y=trainY, family = "binomial")
lasso.fit.cv <- cv.glmnet(x=as.matrix(trainX), y=trainY, 
                          family = "binomial", type.measure = "auc")
plot(lasso.fit.cv)
coef.min <- coef(lasso.fit, lasso.fit.cv$lambda.min)
coef(lasso.fit, lasso.fit.cv$lambda.1se)
coef(lasso.fit, lasso.fit.cv$lambda.min)

lasso.fit.cv$cvm[lasso.fit.cv$lambda==lasso.fit.cv$lambda.min]
lasso.fit.cv$cvm[lasso.fit.cv$lambda==lasso.fit.cv$lambda.1se]
##### Creating model using variables from lasso variable selection 
attach(heart)
fit1<-glm(formula = target ~cp+thalach+exang+oldpeak+ca+thal, family = "binomial", data = train)
summary(fit1)

library(ROCR)
##### Confusion matrix for Lasso Model (Model 1)
pred.prob <- predict(fit1, newdata = train, type = "response")
pred <- prediction(pred.prob, train$target)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
##### In Sample AUC
slot(performance(pred, "auc"), "y.values")[[1]]
##### Out of Sample AUC
pred.prob.test <- predict(fit1, newdata = test, type = "response")
pred <- prediction(pred.prob.test, test$target)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
slot(performance(pred, "auc"), "y.values")[[1]]

pcut<- mean(train$target)
pred.class <- (pred.prob>pcut)*1
table(train$target, pred.class, dnn = c("True", "Pred"))

##### testing sample
pred.class.test <- (pred.prob.test>pcut)*1
table(test$target, pred.class.test, dnn = c("True", "Pred"))

# (equal-weighted) misclassification rate
MR<- mean(test$target!=pred.class.test)
# False positive rate
FPR<- sum(test$target==0 & pred.class.test==1)/sum(test$target==0)
# False negative rate
FNR<-sum(test$target==1 & pred.class.test==0)/sum(test$target==1)
FNR
FPR
MR
###### cv model 1 (Lasso Model )
library(caret)
fit.control <- trainControl(method = "cv", 
                            number = 10, 
                            summaryFunction = twoClassSummary, 
                            classProbs = TRUE)
heart$target<-ifelse(heart$target=="1","Yes","No")

cv.heart.model1 <- train(
  form = target ~cp+thalach+exang+oldpeak+ca+thal, 
  data = heart,
  trControl = fit.control,
  method = "glm",
  family = "binomial"
)
cv.heart.model1

##### in-sample prediction
pred.lasso.train<- predict(lasso.fit, newx=trainX, s=lasso.fit.cv$lambda.1se, type = "response")
###### out-of-sample prediction
pred.lasso.test<- predict(lasso.fit, newx=testX, s=lasso.fit.cv$lambda.1se, type = "response")

sum1<- summary(fit1)
sum1
sum1$deviance/sum1$df.residual
##### Deviance of models
fit1$deviance
AIC(fit1)
BIC(fit1)

#################################################
#Model 2: SVM
# Load in the Data
setwd("/Volumes/DESKTOP/Grad School/BSAN 750/Final Project")
heart <- read.csv("heart.csv")
head(heart)

# Load in Packages
library("dplyr")
library("faux")
library("DataExplorer")
library("caret")
library("randomForest")
library("kernlab")
library("ROCR")
library("ggplot2")
library("pROC")

# Save categorical features as factors
# Center and scale numeric features
heart <- heart %>%
  mutate_at(c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"), 
            as.factor) %>%
  mutate_if(is.numeric, scale)

# Distribution of Target Variable
table(heart$target)
table(heart$target) / length(heart$target)

# Type of variables visual
plot_intro(heart)
# Categorical variables in data set visual
plot_bar(heart)
# Correlation matrix Plot
plot_correlation(heart)

# Define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 5, # number of repeats
                      number = 10) # number of folds

# Features
x <- heart %>%
  select(-target) %>%
  as.data.frame()

# Target variable
y <- heart$target

# Training: 75%; Test: 25%
set.seed(15)
inTrain <- createDataPartition(y, p = .75, list = FALSE)[,1]
x_train <- x[ inTrain, ]
x_test  <- x[-inTrain, ]
y_train <- y[ inTrain]
y_test  <- y[-inTrain]

# Run RFE
result_rfe1 <- rfe(x = x_train, 
                   y = y_train, 
                   sizes = c(1:13),
                   rfeControl = control)
result_rfe1

# Print the selected features
result_rfe1 <- rfe(x = x_train, 
                   y = y_train, 
                   sizes = c(1:8),
                   rfeControl = control)

predictors(result_rfe1)

# Print the results visually
ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()

# Select top 10 instead of 12 for simplicity
predictors(result_rfe1)[1:10]

# Visualize importance for selected variables
varimp_data <- data.frame(feature = row.names(varImp(result_rfe1))[1:10],
                          importance = varImp(result_rfe1)[1:10, 1])
ggplot(data = varimp_data, 
       aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
  geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
  geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
  theme_bw() + theme(legend.position = "none")

# Select top 8 instead of 10 for simplicity
predictors(result_rfe1)[1:8]

# Visualize importance for selected variables
varimp_data <- data.frame(feature = row.names(varImp(result_rfe1))[1:8],
                          importance = varImp(result_rfe1)[1:8, 1])
ggplot(data = varimp_data, 
       aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
  geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
  geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
  theme_bw() + theme(legend.position = "none")

# Post prediction
postResample(predict(result_rfe1, x_test), y_test)

# Take in 3 parameters
trctrl <- trainControl(method="repeatedcv",
                       number = 10,
                       repeats=3)
# Train Method
train <- cbind(x_train,y_train)
svm.linear <- train(as.numeric(y_train)~thal+ca+cp+exang+sex+thalach+slope+oldpeak,
                    data=train,
                    method = "svmLinear",
                    trControl = trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm.linear[4] #RMSE

svm.linear1 <- train(y_train~thal+ca+cp+exang+sex+thalach+slope+oldpeak,
                     data=train,
                     method = "svmLinear",
                     trControl = trctrl,
                     preProcess = c("center", "scale"),
                     tuneLength = 10)
svm.linear1[4] #Accuracy

# Confusion Matrix
test <- cbind(x_test,y_test)
test_pred <- predict(svm.linear1, newdata = test)
test_pred <- as.numeric(test_pred)-1
confusionMatrix(table(test_pred, test$y_test))

# Whole data set
svm.linear <- train(as.numeric(target)~thal+ca+cp+exang+sex+thalach+slope+oldpeak,
                    data=heart,
                    method = "svmLinear",
                    trControl = trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm.linear[4] #RMSE

svm.linear1 <- train(target~thal+ca+cp+exang+sex+thalach+slope+oldpeak,
                     data=heart,
                     method = "svmLinear",
                     trControl = trctrl,
                     preProcess = c("center", "scale"),
                     tuneLength = 10)
svm.linear1[4] #Accuracy

# AUC
svm1 <- svm(y_train~thal+ca+cp+exang+sex+thalach+slope+oldpeak, data = train)
train.svm.pred <- predict(svm1, test)
roc_svm_test <- roc(response = test$y_test, predictor =as.numeric(train.svm.pred))
plot(roc_svm_test,col = "green2", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)
legend(0.3, 0.2, legend = c("test-svm"), lty = c(1), col = c("blue"))

# MR
MR = 1-svm.linear1$results[2]
names(MR)[1] <- 'Misclassification Rate'
MR

###################################################
#Model 3 Random Forest
heart<-read.csv("heart.csv")
str(heart)
index<- sample(nrow(heart),nrow(heart)*0.75)
train<-heart[index,]
test<-heart[-index,]
heart$target<-ifelse(heart$target=="1","Yes","No")

#random forest
library(randomForest)
heart.rf<- randomForest(as.factor(target)~., data = train, cutoff=c(4/5,1/5)) 
heart.rf

#we are plotting the error rate vs. ntree
plot(heart.rf, lwd=rep(2, 3))
legend("right", legend = c("OOB Error", "FPR", "FNR"), lwd=rep(2, 3), lty = c(1,2,3), col = c("black", "red", "green"))

#prediction 
heart.rf.pred<- predict(heart.rf, newdata=test, type = "prob")[,2]
library(ROCR)
pred <- prediction(heart.rf.pred, test$target)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#AUC number 
unlist(slot(performance(pred, "auc"), "y.values"))

#the confusion matrix based on class prediction
heart.rf.class.test<- predict(heart.rf, newdata=test, type = "class")
table(test$target, heart.rf.class.test, dnn = c("True", "Pred"))

#the confusion matrix based on probabilistic prediction with user specified cutoff (overall default rate).

heart.rf.class.test<- (heart.rf.pred>mean(train$target))*1
table(test$target, heart.rf.class.test, dnn = c("True", "Pred"))

MR<- mean(test$target!=heart.rf.class.test)
MR
FPR<- sum(test$target==0 & heart.rf.class.test==1)/sum(test$target==0)
FNR<-sum(test$target==1 & heart.rf.class.test==0)/sum(test$target==1)
FPR
FNR
#variable importance 
heart.rf$importance

library(vip)
vip(heart.rf, num_features = 15, geom = "point")

#######################################################
#Model 4 Boosting
##### Reading in initial heart data and creating training/test sets
heart<-read.csv("heart.csv")
str(heart)
index<- sample(nrow(heart),nrow(heart)*0.80)
train<-heart[index,]
test<-heart[-index,]
heart$target<-ifelse(heart$target=="1","Yes","No")

###### random forest
install.packages("randomForest")
library(randomForest)
heart.rf<- randomForest(as.factor(target)~., data = train, cutoff=c(4/5,1/5)) 
heart.rf

####### we are plotting the error rate vs. ntree
plot(heart.rf, lwd=rep(2, 3))
legend("right", legend = c("OOB Error", "FPR", "FNR"), lwd=rep(2, 3), lty = c(1,2,3), col = c("black", "red", "green"))

###### prediction 
heart.rf.pred<- predict(heart.rf, newdata=test, type = "prob")[,2]
library(ROCR)
pred <- prediction(heart.rf.pred, test$target)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

##### AUC number 
unlist(slot(performance(pred, "auc"), "y.values"))
##### the confusion matrix based on class prediction
heart.rf.class.test<- predict(heart.rf, newdata=test, type = "class")
table(test$target, heart.rf.class.test, dnn = c("True", "Pred"))
##### the confusion matrix based on probabilistic prediction with user specified cutoff (overall default rate).
heart.rf.class.test<- (heart.rf.pred>mean(train$target))*1
table(test$target, heart.rf.class.test, dnn = c("True", "Pred"))
##### variable importance 
heart.rf$importance
##### More charts
install.packages("vip")
library(vip)
vip(heart.rf, num_features = 15, geom = "point")

#####boosting 
install.packages("gbm")
library(gbm)
heart.boost<- gbm(target~., data = train, distribution = "bernoulli", 
                  n.trees = 5000, cv.folds = 5, n.cores = 5)
summary(heart.boost)

best.iter <- (gbm.perf(heart.boost, method = "cv"))
best.iter

###### predicted probability
pred.heart.boost<- predict(heart.boost, newdata = test, n.trees = best.iter, type="response")
pred.heart.boost
##### AUC
pred <- prediction(pred.heart.boost, test$target)
unlist(slot(performance(pred, "auc"), "y.values"))

pred.heart.boost.class<- (pred.heart.boost>mean(train$target))*1
table(test$target, pred.heart.boost.class, dnn = c("True", "Pred"))

##### Still need to do
# (equal-weighted) misclassification rateMR<- mean(test$target!=pred.heart.boost.class)
MR
##### False positive rate
FPR<- sum(test$target==0 & pred.heart.boost.class==1)/sum(test$target==0)
FPR
##### False negative rate
FNR<-sum(test$target==1 & pred.heart.boost.class==0)/sum(test$target==1)
FNR