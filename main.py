from data_preparation import datasets_to_csv, prepare_data
from input_validation import validate_input, validate_navigation_input
from model_training_evaluation import SalaryPredictor
from visualizations import save_salary_distributions, save_usage_vs_salary, save_correlation_heatmap, open_histogram, \
    open_scatterplot, open_correlation_heatmap, save_feature_importance, open_barchart

# Raw dataset CSV files are cleaned, with dataframes returned that are ready for use
main_dataset, train_dataset, test_dataset = prepare_data('raw_data/22-23 All stats.csv', 'raw_data/22-23 Salaries.csv',
                                                         'raw_data/22-23 Team Payroll.csv',
                                                         'raw_data/23-24 All stats.csv', 'raw_data/23-24 Salaries.csv',
                                                         'raw_data/23-24 Team Payroll.csv',
                                                         'raw_data/24-25 All stats.csv', 'raw_data/24-25 Salaries.csv',
                                                         'raw_data/24-25 Team Payroll.csv')

# Prepared dataframes are converted to CSV to save/overwrite the current cleaned data files to project folder
datasets_to_csv(main_dataset, train_dataset, test_dataset)

# Visualizations are saved as png files. The name of the file is then returned which can be used to open the file later
year3, year2, year1 = main_dataset['Year'].unique()
histogram = save_salary_distributions(main_dataset, year1, year2, year3)
scatterplot = save_usage_vs_salary(main_dataset, year1, year2, year3)
correlation_heatmap = save_correlation_heatmap(main_dataset, year1, year3)

# Pass training and test dataset to model object
# Train Model
NBAmodel = SalaryPredictor(train_dataset, test_dataset)
NBAmodel.train_model()

# Post-training visualization of feature importances
bar_chart = save_feature_importance(train_dataset, NBAmodel)

# Start of User Interface
quit = False
while not quit:
    print('\nNBA Salary Predictor Dashboard\n'
          '\t1: How to use application\n'
          '\t2: Predict Player Salary Now!\n'
          '\t3: Learn about Model\n'
          '\t4: Quit\n')
    user_choice = validate_navigation_input(4)

    if user_choice == 1:
        print("""
How to use application

    This application is intended solely for the use of the Chicago Bulls Organization. The following options are explained below:

        Please Click Option 2 to get started with predicting NBA player salaries. The specific input will be outlined here, and it is 
        the user's responsibility to obtain this required information, whether its existing or projected stats for the given player. 
        Each of the stats will be input by the user, and the recommended salary percentage will then be displayed for the user. 
        Please ensure care and attention to detail when inputting stats, as inadvertently typing a value intended for another stat 
        category will likely alter the results.

        Please Click Option 3 to learn more about the model. Here, some key insights into the model's background will be given. 
        Additionally, the user will be able to see the evaluation metrics used by the development team, and visualizations of 
        the datasets used to create the model. Option 3 serves to give the user better context in understanding how the model works, 
        its performance, and the data behind it.

        Please Click Option 4 to exit the application.

    Note: If there are any further questions or concerns, please contact the Analytics Department at chicagobullsanalytics@nba.com 

Press any key to return home
        """)
        to_dashboard = input()

    if user_choice == 2:
        print("""
Player Salary Prediction

    The following information will be required: 
        Age of Player- enter a whole number (e.g., 25)
        Total Minutes in a Season- enter a whole number (e.g., 25)
        True Shooting Percentage- enter a number, can include decimals (e.g.,43.4)
        Total Rebound Percentage- enter a number, can include decimals (e.g.,43.4)
        Total Assist Percentage- enter a number, can include decimals (e.g.,43.4)
        Total Steals Percentage- enter a number, can include decimals (e.g.,43.4)
        Total Blocks Percentage- enter a number, can include decimals (e.g.,43.4)
        Total Turnover Percentage- enter a number, can include decimals (e.g.,43.4)
        Usage Rate Percentage- enter a number, can include decimals (e.g.,43.4)
        Box Plus/Minus- enter a number, can include decimals (e.g.,43.4)
        Availability Percentage- enter a number, can include decimals (e.g.,43.4)
        """)
        print('Are you ready with this information?\n'
              '1: Yes, Start now\n'
              '2: No, Return to Dashboard\n')
        user_choice2 = validate_navigation_input(2)

        if user_choice2 == 1:
            age = validate_input('Enter the Age of the Player: ', int, 19, 100)
            total_minutes = validate_input('Enter the Total Minutes for the season: ', int, 0, 4000)
            ts = validate_input('Enter the True Shooting Percentage: ', float, 0, 100)
            ts = ts / 100  # True shooting % is commonly reported in decimals, so converting it here to match our dataset
            trb = validate_input('Enter the Total Rebound Percentage: ', float, 0, 100)
            ast = validate_input('Enter the Total Assist Percentage: ', float, 0, 100)
            stl = validate_input('Enter the Total Steal Percentage: ', float, 0, 100)
            blk = validate_input('Enter the Total Block Percentage: ', float, 0, 100)
            tov = validate_input('Enter the Total Turnover Percentage: ', float, 0, 100)
            usg = validate_input('Enter the Total Usage Percentage: ', float, 0, 100)
            bpm = validate_input('Enter the Box Plus/Minus Percentage: ', float, 0, 100)
            avb = validate_input('Enter the Availability Percentage: ', float, 0, 100)

            answers = [age, total_minutes, ts, trb, ast, stl, blk, tov, usg, bpm, avb]
            predicted_value = NBAmodel.predict(answers)

            print(f"\n\tRecommended Salary Percentage (of team payroll): {predicted_value:.2f}% of Team Payroll")
            print('\nTo return home, press any key and hit enter')
            return_home = input()

    if user_choice == 3:
        print("""
Learn about the model

        How is this Prediction being made?
            This prediction is being made behind-the-scenes by a supervised machine learning model. While this sounds fancy, all 
            this means is that our model is provided with historical statistics to analyze, including advanced stats and other 
            key contextual factors like age and minutes played. Afterwards, the models job is to identify complex patterns within 
            these stats and use these relationships to ultimately predict a players salary (in percentage form) based on the data you input.

        Which datasets were used to train the model?
            The most recent datasets used to train the model were from the 2022-2023 and 2023-2024 NBA seasons (excluding playoffs). 
            The model was simulated on the 2024-2025 datasets, with known salary amounts, in order to test the performance

        How do we know the prediction is accurate?
            The model was trained on historical data as mentioned above. Afterwards, it was given a dataset with known salaries of players, 
            along with the same stats used to train the model. The predicted salary values and actual salary values were compared and 
            evaluated using well-known metrics that measured the performance of the model.

        Does the model ever need to be retrained?
            The short answer is yes and no. The model does not require retraining each year because it calculates salary percentage instead 
            of a raw salary amount. Thus, even with the changing landscape of salary cap spaces, the salary percentage can tell the user how 
            much to allocate the player in relation to the teams payroll. However, we will be periodically evaluating methods of optimizing 
            the model, so retraining is always a possibility.
            """)
        print('\nChoose from the following options to learn more or return home:\n'
              '\n\t1: See Evaluation Metrics\n'
              '\t2: See Data Visualizations\n'
              '\t3: Return to Dashboard\n')

        user_choice2 = validate_navigation_input(3)
        if user_choice2 == 1:
            mae_model, r_model = NBAmodel.test_model()
            mae_baseline, r_baseline = NBAmodel.test_baseline_model()
            print(f"""
Evaluation Metrics

        The following metrics were used in analyzing our predictive model: Mean Absolute Error and Coefficient of Determination

                Mean Absolute Error (MAE): This value measures the average prediction error of the salary percentage of each player, between 
                predicted value and actual value. This essentially tells us the level of accuracy of our model's predictive ability.

                Coefficient of Determination (R-squared): Measure how much of the variation in player salary our model can explain using our
                selected data features. This essentially tells us the explanatory power of our model.

        Baseline model: A baseline model was created to establish specific benchmarks that our model needed to outperform in order
        to be considered ready for implementation. This baseline model trained and tested on the same datasets as our trained model, 
        except the predictions were based on the average league salary percentage. Below are the results. 

                                                        Evaluation Results

                                            Metric          Baseline         Our Model
                                            ------------------------------------------
                                            MAE:        {mae_baseline:>10.2f}   {mae_model:>14.2f}
                                            R²:         {r_baseline:>10.2f}     {r_model:>12.2f}

         As shown above, the MAE was decreased by almost half and the R-squared value increased significantly, showing the accuracy 
         and strong explanatory power of our model.

         Press any key to return home
                """)
            to_dashboard = input()

        if user_choice2 == 2:
            print("""
Data Visualizations

        The following visualizations give insight into the datasets that were used to build this ML Salary Model. Press any corresponding key to explore!
            """)
            return_home = False
            while not return_home:
                print('\t1: Salary % distribution of players in the past 3 seasons\n'
                      '\t2: Usage rate % of players vs Salary %\n'
                      '\t3: Correlation heatmap of player statistics and salary\n'
                      '\t4: Feature importance of trained model\n'
                      '\t5: Return To Dashboard')
                user_choice_visualization = validate_navigation_input(5)
                if user_choice_visualization == 1:
                    open_histogram(histogram)
                if user_choice_visualization == 2:
                    open_scatterplot(scatterplot)
                if user_choice_visualization == 3:
                    open_correlation_heatmap(correlation_heatmap)
                if user_choice_visualization == 4:
                    open_barchart(bar_chart)
                if user_choice_visualization == 5:
                    return_home = True
        if user_choice2 == 3:
            quit = False
    if user_choice == 4:
        print('Thank you for using the NBA Salary Prediction Tool!')
        quit = True