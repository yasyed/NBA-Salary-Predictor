import pandas as pd

#Raw dataset files are passed to function and are cleaned up and prepared for use
def prepare_data(season1_stats, season1_salaries, season1_team_payroll,
                 season2_stats, season2_salaries, season2_team_payroll,
                 season3_stats, season3_salaries, season3_team_payroll):

    # CSV files are read for the 2022-2023 Season and merged
    advanced_stats_season1 = pd.read_csv(season1_stats)
    salary_table_season1 = pd.read_csv(season1_salaries)
    team_total_season1 = pd.read_csv(season1_team_payroll)
    dataset_season1 = pd.merge(advanced_stats_season1, salary_table_season1, on='Player')
    dataset_season1 = pd.merge(dataset_season1, team_total_season1, on='Team')

    # CSV files are read for the 2023-2024 Season and merged
    advanced_stats_season2 = pd.read_csv(season2_stats)
    salary_table_season2 = pd.read_csv(season2_salaries)
    team_total_season2 = pd.read_csv(season2_team_payroll)
    dataset_season2 = pd.merge(advanced_stats_season2, salary_table_season2, on='Player')
    dataset_season2 = pd.merge(dataset_season2, team_total_season2, on='Team')

    #CSV files are read for the 2024-2025 Season and merged
    advanced_stats_season3= pd.read_csv(season3_stats)
    salary_table_season3 = pd.read_csv(season3_salaries)
    team_total_season3 = pd.read_csv(season3_team_payroll)
    dataset_season3 = pd.merge(advanced_stats_season3, salary_table_season3,on='Player')
    dataset_season3 = pd.merge(dataset_season3, team_total_season3, on='Team')

    # The 3 datasets are attached to form the main dataset
    main_dataset = pd.concat([dataset_season3,dataset_season2], ignore_index=True)
    main_dataset = pd.concat([main_dataset,dataset_season1], ignore_index=True)

    #Need to remove special symbols like $ and commas
    main_dataset['Team Payroll'] = main_dataset['Team Payroll'].replace('\\$','', regex=True)
    main_dataset['Team Payroll'] = main_dataset['Team Payroll'].replace(',','', regex=True)
    main_dataset['Salary'] = main_dataset['Salary'].replace('\\$','', regex=True)
    main_dataset['Salary'] = main_dataset['Salary'].replace(',','', regex=True)

    #Need to convert both columns from String to float
    main_dataset['Salary'] = main_dataset['Salary'].astype(float)
    main_dataset['Team Payroll'] = main_dataset['Team Payroll'].astype(float)

    #Availability column created to replace Games played. Columns that are redundant/not needed are removed
    main_dataset['Availability%'] = (main_dataset['G'] / 82) * 100
    main_dataset['Availability%'] = main_dataset['Availability%'].round(2)
    columns_to_drop = ['DRB%','ORB%', 'DWS', 'OWS','WS', 'WS/48', 'DBPM', 'OBPM', '3PAr','G','GS','FTr', 'PER', 'VORP']
    main_dataset = main_dataset.drop(columns = columns_to_drop)

    # The target variable column of Salary percentage is created
    main_dataset['Salary%'] = (main_dataset['Salary'] / main_dataset['Team Payroll']) * 100
    main_dataset['Salary%'] = main_dataset['Salary%'].round(2)

    #Separate the trained and test files and convert to CSV
    year3, year2, year1 = main_dataset['Year'].unique()
    train_dataset = main_dataset[main_dataset['Year'].isin([year1, year2])]
    train_dataset = train_dataset.drop(columns = ['Salary', 'Team Payroll', 'Team','Year', 'Player'])

    test_dataset = main_dataset[main_dataset['Year'].isin([year3])]
    test_dataset = test_dataset.drop(columns = ['Salary', 'Team Payroll','Team','Year', 'Player'])

    return main_dataset,train_dataset,test_dataset #DataFrame objects are returned

#Create CSV files for the cleaned datasets (after editing the raw data files)
def datasets_to_csv(dataset1, dataset2, dataset3):
    main_dataset = dataset1
    train_dataset = dataset2
    test_dataset = dataset3
    main_dataset.to_csv('cleaned_data/main_dataset.csv', index = False)
    train_dataset.to_csv('cleaned_data/train_dataset.csv', index = False)
    test_dataset.to_csv('cleaned_data/test_dataset.csv', index = False)





