from Logistic_Regression_Class import Logistic_Regression

regressor = Logistic_Regression(distribution, method, learning_rate, epochs)

# splitting and normalizing with min-max normalization
x_train, y_train, x_val, y_val, x_test, y_test = regressor.split_normalize(filename, val_ratio, test_ratio)

#training the model
regressor.fit(x_train, y_train)

#predicting
predicts = regressor.predict(x_test)

#accuracy score and confusion matrix
ac = regressor.accuracy_score(y_test, predicts)
cm = regressor.confusion_matrix(y_test, predicts)

# F_05, F_1, F_2 scores and False Positive Rate
F_05, F_1, F_2, fpr = regressor.f_metrics()