### Create wrapper function to train xgboost model

xgb_model = function(param, nround){

	print(paste(names(param),param,collapse=" , "))  ## Print the parameters used for model

	## Select number of rounds using xgb.cv
	bst_cv = xgb.cv(param=param, data = train_x, label = train_y, nrounds=nround,nfold=2,metric=list("auc","error"))
	nround_sel = which.max(bst_cv$test.auc.mean)
	print(paste("Nround Selected is :", nround_sel));
	print(bst_cv[nround_sel,]);

	## Train final model using number of rounds selected using xgb.cv
	final_model = xgboost(param=param, data = train_x, label = train_y, nrounds=nround_sel)
	return(final_model)
}
