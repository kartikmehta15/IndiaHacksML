HackerEarth organized "IndiaHacks: Machine Learning Hackathon" from 15/01/2016 to 25/01/2016. This is my solution for this competition which put me in first place on the Publc as well as Private Leaderboard. The link to the competition is https://www.hackerearth.com/machine-learning-india-hacks-2016/machine-learning/will-bill-solve-it/

Introduction :  HackerEarth is a community of programmers. Thousands of hackers solve problems on HackerEarth everyday to improve their programming skills or win prizes. These hackers can be beginners who are new to programming, or experts who know the solution in a blink. There is a pattern to everything, and this problem is about finding those patterns and problem solving behaviours of the users. Finding these patterns will be of immense help to the problem solvers, as it will allow to suggest relevant problems to solve and offer solution when they seem to be stuck. The opportunities are diverse and you are entitled with the task to predict them.

Given data of the submissions, problems, and users, the aim of the competition is to predict whether a user will be able to solve a problem or not. Accuracy is the metric used in the competition.

Codes : All codes are written in R language. o0_model_pipeline.R is the main code which calls o01_create_dataset.R to create dataset and o02_train_xgboost_model.R for xgboost model.

Approach:

1. Label encoded (or converted to factor and then to numeric in R's language) all categorical features (including user_id and problem_id).
2. Created a couple of new variables like Number of skills, User accuracy (use overall average if user has solved less than 5 problems).
3. Trained 5 Xgboost models (depth=5,10,15,25,30) with hyper parameters of each tuned using Random Search.
4. Ensembled these 5 models using linear blending (though ensembling brought only minor lift which was expected also as there is not much diversity among the models).

What didnt't work:

1. Binary encoded skills but that didnt bring any lift.
2. Experimented with interaction features (userid-problemLevel, usertype-problemLevel, problemId-userType, problemId-skill etc) but could not get any gain.
3. Trained a FTRL model (Refer: https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory or https://www.kaggle.com/kartikmehtaiitd/springleaf-marketing-response/stochastic-gradient-descent) which gave leaderboard score of 0.821. Used it as one of the models for ensembling with xgboost but got only negligible gain in cross validation.

What I should/could have tried:

1. One hot encoding (or hashing) categorical features and training diverse models and using them for ensemble.
2. Training diverse xgboost models using bagging or probabilistic feature inclusion.
3. Better feature engineering.
4. Training different models for all 4 combinations of Users (New/Old) with Problems (New/Old). A user/problem is termed new if there is no entry in training data but only in test data otherwise it is termed old.
