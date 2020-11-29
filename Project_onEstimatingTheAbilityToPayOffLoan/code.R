#Submitted by Group 6 Sec A
#Deepak Namdev
#Sourabh Mahajan
#Manas Bhageria 
#Preeti
#Rajat Agrawal
----------------------------------------------------------------------------------------------------

setwd("F:/IIM/Term 4/DA with R/Group Project/LoanStatus 2nd")
LoanData = read.csv("Loan Prediction Data.csv", header = T, strip.white = T, na.strings = "NA")
View(LoanData)
str(LoanData)

sum(complete.cases(LoanData))
sum(is.na(LoanData))

# Data Pre-processing
LoanData$Loan.Status = as.factor(ifelse(LoanData$Loan.Status == "Fully Paid", 1, 0))
LoanData$Term = ifelse(LoanData$Term == "Long Term", 1, 0)
LoanData = LoanData[, -c(1,2)]

# Removing the variables with having more than 50% missing data
LoanData = LoanData[, -9]
sum(is.na(LoanData))
LoanData = na.omit(LoanData)
str(LoanData)

# Creating dummy variables
LoanData$Home.Ownership_HaveMortgage = ifelse(LoanData$Home.Ownership == "HaveMortgage", 1, 0)
LoanData$Home.Ownership_HomeMortgage = ifelse(LoanData$Home.Ownership == "Home Mortgage", 1, 0)
LoanData$Home.Ownership_OwnHome = ifelse(LoanData$Home.Ownership == "Own Home", 1, 0)

LoanData = LoanData[, -7]

LoanData$Bankruptcies = as.integer(LoanData$Bankruptcies)
LoanData$Tax.Liens = as.integer(LoanData$Tax.Liens)


#partition the dataset 
library(caTools)
set.seed(6)
split = sample.split(LoanData[,1], SplitRatio = 0.8)
TrainingDataset <- subset(LoanData,split=="TRUE")
TestingDataset <- subset(LoanData,split=="FALSE")

# Checking the proportion of 0,1 in #complete #training and #testing data 
prop.table(table(LoanData$Loan.Status))
prop.table(table(TrainingDataset$Loan.Status))
prop.table(table(TestingDataset$Loan.Status))

#Logistic regression model
lr1 = glm(TrainingDataset$Loan.Status ~.,data = TrainingDataset, family = binomial)
summary(lr1)

# Overall Model Significance
with(lr1,null.deviance-deviance)
with(lr1,df.null-df.residual)
#p-value of the test
with(lr1,pchisq(null.deviance-deviance,df.null-df.residual,lower.tail=FALSE))

#check for assumptions 
TrainingPredictedProb = predict(lr1, newdata = TrainingDataset, type = "response")
logit_TrainingDataset = log((TrainingPredictedProb)/(1- TrainingPredictedProb))

library(ggplot2)
library(ggpubr)

a = ggplot(TrainingDataset, aes(logit_TrainingDataset, Current.Loan.Amount))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
b = ggplot(TrainingDataset, aes(logit_TrainingDataset,Credit.Score ))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
c = ggplot(TrainingDataset, aes(logit_TrainingDataset, Annual.Income))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
d = ggplot(TrainingDataset, aes(logit_TrainingDataset, Monthly.Debt))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
e = ggplot(TrainingDataset, aes(logit_TrainingDataset, Years.of.Credit.History))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
f = ggplot(TrainingDataset, aes(logit_TrainingDataset, Number.of.Open.Accounts))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
g = ggplot(TrainingDataset, aes(logit_TrainingDataset, Number.of.Credit.Problems))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
h = ggplot(TrainingDataset, aes(logit_TrainingDataset, Current.Credit.Balance))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()
i = ggplot(TrainingDataset, aes(logit_TrainingDataset, Maximum.Open.Credit))+
  geom_point(size = 0.5, alpha = 0.5) + geom_smooth(method = "loess")  + theme_bw()

ggarrange(a,b,c,d,e,f,g,h,i)


#multicolinearity
library(car)
vif(lr1)

ReducedModel <- glm(TrainingDataset$Loan.Status ~ 1,data = TrainingDataset, family = binomial) 
FullModel <- glm(TrainingDataset$Loan.Status ~. ,data = TrainingDataset, family = binomial) 
x <- step(ReducedModel,scope=list(lower=ReducedModel,upper=FullModel),direction = "both",
          test="Chisq",trace=TRUE)

#SIGNIFICANT MODEL
SignificantModel = glm(TrainingDataset$Loan.Status ~ Current.Loan.Amount + Annual.Income + 
  Monthly.Debt + Term + Home.Ownership_HomeMortgage + Years.in.current.job , 
  data = TrainingDataset, family = binomial)
summary(SignificantModel)
vif(SignificantModel)


#Prediction using LR1
PredictedProb_LR1 = predict(lr1, newdata = TestingDataset, type = "response")
library(InformationValue)
ThresholdProb_LR1 = optimalCutoff(actuals = TestingDataset$Loan.Status, predictedScores = PredictedProb_LR1, 
                              optimiseFor = "Ones")
PredictedLoanStatus_LR1 = ifelse(PredictedProb_LR1 > ThresholdProb_LR1, 1, 0)
p_LR1 = table(ActualValue = TestingDataset$Loan.Status, predictedValue = PredictedLoanStatus_LR1)
accuracy_LR1 <- (p_LR1[1,1]+p_LR1[2,2])/sum(p_LR1)
accuracy_LR1

#Prediction using SIGNIFICANT MODEL
PredictedProb_SM = predict(SignificantModel, newdata = TestingDataset, type = "response")
library(InformationValue)
ThresholdProb_SM = optimalCutoff(actuals = TestingDataset$Loan.Status, predictedScores = PredictedProb_SM, 
                              optimiseFor = "Ones")
PredictedLoanStatus_SM = ifelse(PredictedProb_SM > ThresholdProb_SM, 1, 0)
p_SM = table(ActualValue = TestingDataset$Loan.Status, predictedValue = PredictedLoanStatus_SM)
accuracy_SM = (p_SM[1,1]+p_SM[2,2])/sum(p_SM)
accuracy_SM

#ROC curve LR1 model
library(InformationValue)
plotROC(actuals = TestingDataset$Loan.Status, predictedScores = PredictedProb_LR1)

#ROC curve SIGNIFICANT MODEL
library(InformationValue)
plotROC(actuals = TestingDataset$Loan.Status, predictedScores = PredictedProb_SM)

### Decision Tree ###

library(rpart)
Loanfit <- rpart(Loan.Status~ .,data=TrainingDataset, method="class",control = rpart.control(minsplit=1,cp=0))
print(Loanfit)

# output of rpart.plot can be customised with different parameters
library(rpart.plot)
rpart.plot(Loanfit,main ="Classification Tree")

#error on the test data
Loan.predict <- predict(Loanfit, TestingDataset, type = "class")

# confusion matrix (testing data)
conf.matrix <- table(TestingDataset$Loan.Status, Loan.predict)
print(conf.matrix)

#Accuracy
(conf.matrix[1,1]+conf.matrix[2,2])/sum(conf.matrix)

## Tree Pruning ##

printcp(Loanfit)
ltree <- prune(Loanfit, cp= 0.00215)

rpart.plot(ltree,main ="Pruned Tree")

#error on the test data after pruning
prune.Loan.predict <- predict(ltree, TestingDataset, type = "class")

# confusion matrix (testing data)
conf.matrix <- table(TestingDataset$Loan.Status, prune.Loan.predict)
print(conf.matrix)

#Accuracy
(conf.matrix[1,1]+conf.matrix[2,2])/sum(conf.matrix)


### Random Forest ###

library(randomForest)

#predictor variables at each split and planting 5000 trees for analysis.
Loanfit.rf <- randomForest(Loan.Status ~ .,data=TrainingDataset, mtry=4, ntree= 500)

str(TrainingDataset)

# Performance of tree after random forest
loan.rf <- predict(Loanfit.rf, TestingDataset, type = "class")

# confusion matrix (testing data)
conf.matrix.rf <- table(TestingDataset$Loan.Status, loan.rf)
rownames(conf.matrix.rf) <- paste("Actual", rownames(conf.matrix.rf), sep = ":")
colnames(conf.matrix.rf) <- paste("Predicted", colnames(conf.matrix.rf), sep = ":")
print(conf.matrix.rf)

#Accuracy
(conf.matrix.rf[1,1]+conf.matrix.rf[2,2])/sum(conf.matrix.rf)


