setwd("~/Downloads/Personal/Hackathon/Experiments/")
library(xgboost)

source("o02_train_xgboost_model.R")
source("o01_create_dataset.R")


## Set parameters for 5 xgboost models of different depths. Parameters for each depth have been selected using Random Search.
param1 <- list("objective" = "binary:logistic","eval_metric" = "error","nthread" = 8,max.depth = 25,
              verbose=1,eta = 0.12,gamma=0.1,colsample_bytree=0.8,min_child_weight=50,silent=0)

param2 <- list("objective" = "binary:logistic","eval_metric" = "error","nthread" = 8,max.depth = 30,
                verbose=1,eta = 0.05,gamma=1,colsample_bytree=0.8,min_child_weight=50,silent=0)

param3 <- list("objective" = "binary:logistic","eval_metric" = "error","nthread" = 8,max.depth = 5,
               verbose=1,eta = 0.2,gamma=0,colsample_bytree=0.9,min_child_weight=50,silent=0)

param4 <- list("objective" = "binary:logistic","eval_metric" = "error","nthread" = 8,max.depth = 10,
               verbose=1,eta = 0.1,gamma=0.1,colsample_bytree=1.0,min_child_weight=20,silent=0)

param5 <- list("objective" = "binary:logistic","eval_metric" = "error","nthread" = 8,max.depth = 15,
               verbose=1,eta = 0.1,gamma=0.1,colsample_bytree=0.8,min_child_weight=50,silent=0)

### Train 5 xgboost models with number of trees selected using cross validation
xgb1 = xgb_model(param1, nround=200)
xgb2 = xgb_model(param2, nround=300)
xgb3 = xgb_model(param3, nround=500)
xgb4 = xgb_model(param4, nround=250)
xgb5 = xgb_model(param5, nround=250)

## Split validation data into two equal parts and use first part for learning the ensemble and second part for validating the ensemble
set.seed(1)
rand2 = runif(nrow(val_x))
val1_x = val_x[rand2<=0.5,]
val2_x = val_x[rand2>0.5,]

val1_y = val_y[rand2<=0.5]
val2_y = val_y[rand2>0.5]

pred_val1_1 = predict(xgb1,val1_x)
pred_val2_1 = predict(xgb2,val1_x)
pred_val3_1 = predict(xgb3,val1_x)
pred_val4_1 = predict(xgb4,val1_x)
pred_val5_1 = predict(xgb5,val1_x)

## Below function learns the linear ensemble optimizing on AUC
init_w = c(1,0,0,0)
val_auc_ensemble <- function(w){
  pred_val_comb = w[1]*pred_val1_1 + w[2]*pred_val2_1 + w[3]*pred_val3_1 + w[4]*pred_val4_1 + (1-w[1]-w[2]-w[3]-w[4])*pred_val5_1
  val_auc = colAUC(pred_val_comb,val1_y)
  return(val_auc)
}
result = optim(val_auc_ensemble,par=init_w, control=list(trace=T,fnscale=-1))

## Check Ensemble model's performance on remaining validation data
pred_val1_2 = predict(xgb1,val2_x)
pred_val2_2 = predict(xgb2,val2_x)
pred_val3_2 = predict(xgb3,val2_x)
pred_val4_2 = predict(xgb4,val2_x)
pred_val5_2 = predict(xgb5,val2_x)
pred_val_comb = result$par[1]*pred_val1_2 + result$par[2]*pred_val2_2 + result$par[3]*pred_val3_2 +result$par[4]*pred_val4_2 + (1-result$par[1]-result$par[2]--result$par[3]--result$par[4])*pred_val5_2

print(colAUC(pred_val1_2,val2_y))
print(colAUC(pred_val2_2,val2_y))
print(colAUC(pred_val3_2,val2_y))
print(colAUC(pred_val4_2,val2_y))
print(colAUC(pred_val5_2,val2_y))
print(colAUC(pred_val_comb,val2_y))

#print(colAUC(pred_val1_2,val2_y))
#0 vs. 1 0.8978586
#> print(colAUC(pred_val2_2,val2_y))
#0 vs. 1 0.8991516
#> print(colAUC(pred_val3_2,val2_y))
#0 vs. 1 0.8983628
#> print(colAUC(pred_val4_2,val2_y))
#0 vs. 1 0.8985821
#> print(colAUC(pred_val5_2,val2_y))
#0 vs. 1 0.8986579
#> print(colAUC(pred_val_comb,val2_y))
#0 vs. 1 0.8994676

## Create Test Predictions for submission
pred_test1 = predict(xgb1,test_x)
pred_test2 = predict(xgb2,test_x)
pred_test3 = predict(xgb3,test_x)
pred_test4 = predict(xgb4,test_x)
pred_test5 = predict(xgb5,test_x)

pred_test_comb = result$par[1]*pred_test1 + result$par[2]*pred_test2 + result$par[3]*pred_test3 + result$par[4]*pred_test4 + (1-result$par[1]-result$par[2]-result$par[3]-result$par[4])*pred_test5
pred_testclass = (pred_test_comb > quantile(pred_test_comb,prob=.44))*1  # Convert probabilities to class

subm_xgboost = data.frame(cbind(test_Ids,pred_testclass)) 
names(subm_xgboost) = c("Id","solved_status")

## merge_trte is coming from o01_create_dataset.R
subm_xgboost[merge_trte$rownum,2] = merge_trte$solved_status ## For user and problem combination in both training and test data, use solved_status from the training.
write.csv(subm_xgboost,"xgboost_ensemble_final.csv",row.names=F,quote=FALSE)
