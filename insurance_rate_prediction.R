rm(list=ls())
setwd("C:/Users/archi/Google Drive/Kinjal MS/Study/Sem1Flex2/DAM")
insurance_data <- read.csv(file="./datasets/insurance.csv", header=TRUE, sep=",")

#include libraries
install.packages("dplyr")
install.packages("psych")
install.packages("car")
install.packages("leaps")
install.packages("MASS")
install.packages("caret")
library(dplyr)
library(psych)
library(car)
library(leaps)
library(MASS)
library(caret)

#Renaming variable children to "number of dependents" for better description
names(insurance_data)[4] <- "NoofDependents"
names(insurance_data)

#Exploratory data analysis

#Take a look at the data
str(insurance_data)
head(insurance_data)
summary(insurance_data)

plot(insurance_data$age, insurance_data$charges,pch=20)
plot(insurance_data$charges,insurance_data$age,pch=20)
plot(insurance_data$sex, insurance_data$charges,pch=20)
plot(insurance_data$bmi,insurance_data$charges,pch=20)
plot(insurance_data$NoofDependents,insurance_data$charges,pch=20)
plot(insurance_data$smoker,insurance_data$charges,pch=20)
plot(insurance_data$region,insurance_data$charges,pch=20)
pairs (insurance_data,pch=20)
pairs.panels(insurance_data[c("age","bmi","NoofDependents","charges")])

#histograms
hist(insurance_data$bmi)

#FEATURE ENGINEERING

#Check for missing values
sum(is.na(insurance_data))

#Create indicator variables for all categorical data
insurance_data$sex=ifelse(insurance_data$sex=="female",1,0)
insurance_data$smoker=ifelse(insurance_data$smoker=="yes",1,0)
insurance_data$sw=ifelse(insurance_data$region=="southwest",1,0)
insurance_data$ne=ifelse(insurance_data$region=="northeast",1,0)
insurance_data$nw=ifelse(insurance_data$region=="northwest",1,0)


#drop categorical variables for which indicator variables have been created
insurance_data <- subset(insurance_data, select=-c(region))

#Divide BMI data into two groups, <30 and >30
insurance_data$bmi30 <- ifelse(insurance_data$bmi >= 30, 1, 0)

#Divide age data into three groups, 18-35, 36-55, >55
insurance_data$age55 <- ifelse(insurance_data$age > 55, 1, 0)
insurance_data$age3655 <- ifelse(insurance_data$age > 35 & insurance_data$age <= 55, 1, 0)


# Split the data into training and test data with 80:20 ratio
train<-sample_frac(insurance_data, 0.8)
sid<-as.numeric(rownames(train)) # because rownames() returns character
test<-insurance_data[-sid,]


#create model with training data
model_insurance <- lm(charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw, data=train)
summary(model_insurance)
par(mfrow = c(2, 2))
plot(model_insurance)

#Calculate SST, SSR and SSRes

SST = sum((train$charges-mean(train$charges))^2)
SSRes = sum((train$charges - model_insurance$fitted.values)^2)
SSR = sum((model_insurance$fitted.values - mean(train$charges))^2)
cat("SST = " , SST)
cat("SSR + SSRes" , (SSR + SSRes)) # =SST

Rsquare = SSR/SST
cat("R-square = ", Rsquare)

#Divide BMI data into two groups, <30 and >30
insurance_data$bmi30 <- ifelse(insurance_data$bmi >= 30, 1, 0)

#Divide age data into three groups, 18-35, 36-55, >55
insurance_data$age55 <- ifelse(insurance_data$age > 55, 1, 0)
insurance_data$age3655 <- ifelse(insurance_data$age > 35 & insurance_data$age <= 55, 1, 0)

#Create new training and test data based on above added variables
#train2<-sample_frac(insurance_data, 0.8)
#dim(train2)
#sid<-as.numeric(rownames(train2)) # because rownames() returns character
#test2<-insurance_data[-sid,]

#create new model with added features and removing regions
model2_insurance <- lm(charges~age55+age3655+sex+bmi+NoofDependents+smoker+bmi30, data=train)
summary(model2_insurance)

#create new model by adding interaction between bmi30 and smoker 
model3_insurance <- lm(charges~age+sex+bmi+NoofDependents+smoker+bmi30*smoker, data=train)
summary(model3_insurance)

# obtain MSRes
MSRes = (summary(model3_insurance)$sigma)^2

# obtain standardized residuals
standardized_res=model3_insurance$residuals/sqrt(MSRes)

# PRESS residuals
PRESS_res=model3_insurance$residuals/(1 - lm.influence(model3_insurance)$hat)
summary(PRESS_res)

# studentized residuals
studentized_res=model3_insurance$residuals/sqrt(MSRes*(1-lm.influence(model3_insurance)$hat))

#Check for data points with high residuals
studentized_res[studentized_res>3 | studentized_res<(-3)]

#Check for data points with high leverage
influence = lm.influence(model3_insurance)$hat
summary(influence) #no high influence points


#Check for non-constancy of error variance with each covariate
par(mfrow = c(1,1))
qqnorm(model3_insurance$residuals)

#Check for non-linearity between covariates and response variable
pairs(train[c("age", "sex", "bmi", "NoofDependents", "smoker", "bmi30", "charges")])
plot(x=train$age, y=model3_insurance$residuals, pch=20)
plot(x=train$bmi+0.3*rnorm(10), y=model3_insurance$residuals+0.3*rnorm(10), pch=20)
plot(x=train$NoofDependents+0.3*rnorm(10), y=model3_insurance$residuals+0.3*rnorm(10), pch=20)
plot(x=train$ind_sex+0.7*rnorm(10), y=model3_insurance$residuals+0.7*rnorm(10), pch=20)
plot(x=train$ind_smoker+0.5*rnorm(10), y=model3_insurance$residuals+0.5*rnorm(10), pch=20)
plot(x=train$sw+0.5*rnorm(10), y=model3_insurance$residuals+0.5*rnorm(10), pch=20)
plot(x=train$se+0.5*rnorm(10), y=model3_insurance$residuals+0.5*rnorm(10), pch=20)
plot(x=train$nw+0.5*rnorm(10), y=model3_insurance$residuals+0.5*rnorm(10), pch=20)
plot(x=train$ne+0.5*rnorm(10), y=model3_insurance$residuals+0.5*rnorm(10), pch=20)


#Plot fitted values vs Residual
plot(x=model3_insurance$fitted.values, model_insurance$residuals, xlab = "Fitted Values", ylab = "Residuals", pch=20)


temp$log_age <- log(temp$age)
head(temp)

#copy train data to temp dataframe and see effects of various transformations
temp <- train
#sqrt transformation
temp$sqrt_charges <- sqrt(temp$charges)
par(mfrow = c(2,2))
hist(temp$charges)
hist(temp$sqrt_charges)
#cube-root transofrmation
temp$cube_charges <- (temp$charges)^(1/3)
hist(temp$cube_charges)
#log transformation
temp$logcharges <- log(temp$charges)
hist(temp$logcharges)

#Check if boxcox suggests any transformations on response variable
require(MASS)
par(mfrow = c(1, 1))
boxcox(model3_insurance) 

#model with cube-root transformation on y
train$cuberoot_charges = (train$charges)^(1/3)
train$logcharges = log(train$charges)

#create new model with cube-root transformed response variable
model_transformed <- lm(cuberoot_charges~age+sex+bmi+NoofDependents+smoker+bmi30*smoker, data=train)
summary(model_transformed)
par(mfrow = c(2, 2))
plot(model_transformed)

#create new model with log transformed response variable
model1_transformed <- lm(logcharges~age+sex+bmi+NoofDependents+smoker+bmi30*smoker, data=train)
summary(model1_transformed)
par(mfrow = c(2, 2))
plot(model1_transformed)

#train2$sqrt_charges = sqrt(train2$charges)
#test2$sqrt_charges = sqrt(test2$charges)

#Tranformation is reducing the Adjusted R-square and not improving normality or equality of variance
#So, not going for transformaed model

insurance_data <- subset(insurance_data, select=-c(sqrt_charges))
train2 <- subset(train2, select=-c(sqrt_charges))
test2 <- subset(test2, select=-c(sqrt_charges))

#check for multicollinearity
vif(model3_insurance)

#Finding the best subset of regressors
#allmodels <- regsubsets(charges~age+ind_sex+bmi+NoofDependents+ind_smoker+bmi30*ind_smoker,data=train2,nbest=3)
#summary(allmodels)
#plot(allmodels)

#Using stepwise regression for covariate selection
add1(lm(charges~1,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added age
add1(lm(charges~age,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added smoker
add1(lm(charges~age+smoker,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added bmi
add1(lm(charges~age+smoker+bmi,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added interaction between bmi30 and smoker
add1(lm(charges~age+smoker+bmi+bmi30*smoker,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added NoofDependents
add1(lm(charges~age+smoker+bmi+bmi30*smoker+NoofDependents,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added age3655
add1(lm(charges~age+smoker+bmi+bmi30*smoker+NoofDependents+age3655,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added sw
add1(lm(charges~age+smoker+bmi+bmi30*smoker+NoofDependents+age3655+sw,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")
#added "sex"
add1(lm(charges~age+smoker+bmi+bmi30*smoker+NoofDependents+age3655+sw+sex,data=train), charges~age+sex+bmi+NoofDependents+smoker+sw+ne+nw+age55+age3655+bmi30*smoker, test="F")

#We will stop adding variables they are not

#Now lets start with removing variable as per the least significance
drop1(lm(charges~age+smoker+bmi+bmi30*smoker+NoofDependents+age3655+sw+sex,data=train), test="F")
#removed bmi
drop1(lm(charges~age+smoker+bmi30*smoker+NoofDependents+age3655+sw+sex,data=train), test="F")
#We will stop here because all the variables are highly significant now

model_final <- lm(charges~age+smoker+bmi30*smoker+NoofDependents+age3655+sw+sex,data=train)
summary(model_final)
plot(model_final)

vif(model_final)

#training the final model using cross-validation
model_final <- train(
  charges~age+smoker+bmi30*smoker+NoofDependents+age3655+sw+sex, train,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE
  )
)
head(test)
head(train)

#Predict results on test data
predicted_test <- predict(model_final, test)
plot(x=test$charges, y=predicted_test, xlab="Actual", ylab="Predicted", col="blue")

plot(model_final)
qqnorm(model_final$residuals)

