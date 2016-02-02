setwd("~/Downloads/Personal/Hackathon/raw_data/")

set.seed(1)
library(tm)
library(data.table)
library(caTools)
require(xgboost)

count_skills = function(x){
  y = length(strsplit(x,"|",fixed=TRUE)[[1]])
  return(y)
}

## Merge training files ##
tr_users = fread("train/users.csv",stringsAsFactors = F)
tr_problems = fread("train/problems.csv",stringsAsFactors = F)
tr_subm = fread("train/submissions.csv",stringsAsFactors = F)

## Create Variable for user accuracy and use global accuracy if user has solved less than 5 problems
tr_users$us_accuracy = tr_users$solved_count/tr_users$attempts 
overall_resp = sum(tr_users$solved_count)/sum(tr_users$attempts)
tr_users$us_accuracy = ifelse(tr_users$solved_count<5,overall_resp,tr_users$us_accuracy)


## Merge Users, problems and submissions to create master training file
traindata1_1 = merge(tr_problems,tr_subm,by="problem_id")
traindata1_2 = merge(traindata1_1,tr_users, by="user_id")
traindata1_2[is.na(traindata1_2)] = 0

traindata1_3 = traindata1_2[solved_status!="UK",c(1:14,17:21),with=F]
traindata1_3$solved_status = ifelse(traindata1_3$solved_status=="SO",1,0)
traindata1_3$numSkills = sapply(traindata1_3$skills, count_skills)
traindata1_3 = traindata1_3[,c(1:13,15:20),with=F]

traindata = traindata1_3[,list(solved_status=max(solved_status)),by=list(user_id,problem_id,level,accuracy,solved_count.x,error_count,rating,tag1,tag2,tag3,tag4,tag5,skills,solved_count.y,attempts,user_type,numSkills,us_accuracy)]


## Merge User, problems and submissions to create master test file
test_users = fread("test/users.csv",stringsAsFactors = F)
test_problems = fread("test/problems.csv",stringsAsFactors = F)
test_subm = fread("test/test.csv",stringsAsFactors = F)

test_users$us_accuracy = test_users$solved_count/test_users$attempts
test_users$us_accuracy = ifelse(test_users$solved_count<5,overall_resp,test_users$us_accuracy)

testdata1_1 = merge(test_problems,test_subm,by="problem_id")
testdata1_2 = merge(testdata1_1,test_users, by="user_id")
testdata1_2[is.na(testdata1_2)] = 0
testdata1_2$numSkills = sapply(testdata1_2$skills, count_skills)
testdata = testdata1_2


############# Combine Train and Test data and Label Encode string/factor variables ###############
test_Ids = testdata$Id
testdata$solved_status = -1
modeldata = data.frame(rbind(traindata,testdata[,names(traindata),with=F]))

## Convert variables to numeric as xgboost only accepts numeric inputs
factor_cols=c("user_id","problem_id","level","tag1","tag2","tag3","tag4","tag5","skills","user_type")
all_cols = setdiff(colnames(modeldata),"")
modeldata[ , factor_cols] <- lapply(modeldata[ , factor_cols] , factor)
modeldata[ , all_cols] <- lapply(modeldata[ , all_cols] , as.numeric)


testdata2 = modeldata[modeldata$solved_status==-1,]
traindata2 = modeldata[modeldata$solved_status>-1,]

# Split training data into 80% training and 20% validation
rand1 = runif(length(traindata2$user_id))
train = traindata2[rand1<=0.8,]
val = traindata2[rand1>0.8,]

train_x = as.matrix(train[,-c(19)])
val_x = as.matrix(val[,-c(19)])
test_x = as.matrix(testdata2[,-c(19)])

train_y = as.numeric(train$solved_status)
val_y = as.numeric(val$solved_status)
### train_x, val_x, test_x, train_y, val_y are the datasets that we need for modeling.

### Find user and problem pairs which are present in both training and test data
tr_us = data.frame(traindata[,c(1,2,19),with=F])
te_us = data.frame(testdata[,c(1,2,13),with=F])
tr_us$up = paste(tr_us$user_id, tr_us$problem_id, sep="-")
te_us$up = paste(te_us$user_id, te_us$problem_id, sep="-")
te_us$rownum = c(1:nrow(testdata))

tr_us = tr_us[,c(3,4)]
te_us = te_us[,c(3,4,5)]

merge_trte = merge(tr_us,te_us)
