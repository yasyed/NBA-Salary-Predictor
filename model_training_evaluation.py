import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

#Model Class, consisting of functions to train, test, and predict with model
class SalaryPredictor:
    def __init__(self, train_dataset, test_dataset):
        self.model = None
        self.features_train = train_dataset.drop(columns = ['Salary%'])
        self.target_train = train_dataset['Salary%']
        self.features_test = test_dataset.drop(columns = ['Salary%'])
        self.target_test = test_dataset['Salary%']

#Model trained on passed feature and target datasets
    def train_model(self):
        self.model = RandomForestRegressor(n_estimators=300, random_state=23)
        self.model.fit(self.features_train, self.target_train)

#Model can predict, once given a list of user inputs
    def predict(self, answers):
        columns = self.features_train.columns
        input_dataframe = pd.DataFrame([answers], columns = columns)
        predicted_value = self.model.predict(input_dataframe) # a numpy array is returned and there is one element of the predicted value
        predicted_value = predicted_value[0] # the predicted value is extracted from the array
        return predicted_value

    #Model tested with results returned
    def test_model(self):
        test_predictions = self.model.predict(self.features_test)
        mae_model = mean_absolute_error(self.target_test, test_predictions)
        r_model = r2_score(self.target_test, test_predictions)
        return mae_model, r_model

    #Baseline model predicts average salary% for all player entries in testing set (with average calculated from training set)
    def test_baseline_model(self):
        average_salary_percentage = self.target_train.mean()
        baseline_target_predicted = [average_salary_percentage] * len(self.target_test)
        mae_baseline = mean_absolute_error(self.target_test, baseline_target_predicted)
        r_baseline = r2_score(self.target_test, baseline_target_predicted)
        return mae_baseline, r_baseline