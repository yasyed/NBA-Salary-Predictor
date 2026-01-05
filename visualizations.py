import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np

#Datasets can be organized by individual season
def get_season_datasets(dataset, year_1, year_2, year_3):
    cleaned_dataset_1 = dataset[dataset['Year'].isin([year_1])]
    cleaned_dataset_2 = dataset[dataset['Year'].isin([year_2])]
    cleaned_dataset_3 = dataset[dataset['Year'].isin([year_3])]
    return cleaned_dataset_1, cleaned_dataset_2, cleaned_dataset_3

# Post-training feature importance bar chart(sorted), saved to folder
def save_feature_importance(train_dataset, model_name):
    feature_variables = train_dataset.columns
    feature_importances = model_name.model.feature_importances_
    ordered_indices = np.argsort(feature_importances)
    plt.figure(figsize=(15,6))
    plt.barh(feature_variables[ordered_indices], feature_importances[ordered_indices], color='blue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance of Model')
    filename = 'visualizations/feature_importance.png'
    plt.savefig(filename)
    plt.close()
    return filename

#The next 3 functions below create the data pre-training visuals and save to project folder
def save_salary_distributions(dataset, year_1, year_2, year_3):
    cleaned_dataset_1, cleaned_dataset_2, cleaned_dataset_3 = get_season_datasets(dataset, year_1, year_2, year_3)

    fig, axes = plt.subplots(1,3, figsize=(15,6))
    axes[0].hist(cleaned_dataset_1['Salary%'],bins=12,color='red', edgecolor='black')
    axes[0].set_title(f'NBA Salary Distribution for {year_1} Season')
    axes[0].set_xlabel('Salary(% of Team Payroll)')
    axes[0].set_ylabel('Number of Players with Salary')
    axes[0].margins(x=0)

    axes[1].hist(cleaned_dataset_2['Salary%'], bins=12, color='red', edgecolor='black')
    axes[1].set_title(f'NBA Salary Distribution for {year_2} Season')
    axes[1].set_xlabel('Salary(% of Team Payroll)')
    axes[1].set_ylabel('Number of Players with Salary')
    axes[1].margins(x=0)

    axes[2].hist(cleaned_dataset_3['Salary%'], bins=12, color='red', edgecolor='black')
    axes[2].set_title(f'NBA Salary Distribution for {year_3} Season')
    axes[2].set_xlabel('Salary(% of Team Payroll)')
    axes[2].set_ylabel('Number of Players with Salary')
    axes[2].margins(x=0)
    filename = 'visualizations/salary_distributions.png'
    plt.savefig(filename)
    plt.close(fig)
    return filename

def save_usage_vs_salary(dataset, year_1, year_2, year_3):
    cleaned_dataset_1, cleaned_dataset_2, cleaned_dataset_3 = get_season_datasets(dataset, year_1, year_2, year_3)

    fig, axes = plt.subplots(1,3, figsize=(15,6))
    axes[0].scatter(cleaned_dataset_1['USG%'], cleaned_dataset_1['Salary%'], color='red', edgecolor='black')
    axes[0].set_title(f'Usage Rate vs Salary for {year_1} Season')
    axes[0].set_xlabel('Usage Rate %')
    axes[0].set_ylabel('Salary Percentage')

    axes[1].scatter(cleaned_dataset_2['USG%'], cleaned_dataset_2['Salary%'], color='red', edgecolor='black')
    axes[1].set_title(f'Usage Rate vs Salary for {year_2} Season')
    axes[1].set_xlabel('Usage Rate %')
    axes[1].set_ylabel('Salary Percentage')

    axes[2].scatter(cleaned_dataset_3['USG%'], cleaned_dataset_3['Salary%'], color='red', edgecolor='black')
    axes[2].set_title(f'Usage Rate vs Salary for {year_3}Season')
    axes[2].set_xlabel('Usage Rate %')
    axes[2].set_ylabel('Salary Percentage')
    filename = 'visualizations/usage_vs_salary.png'
    plt.savefig(filename)
    plt.close(fig)
    return filename

def save_correlation_heatmap(dataset, year_1, year_3):
    drop_columns = ['Team', 'Player', 'Year', 'Salary', 'Team Payroll']
    correlation_matrix = dataset.drop(columns=drop_columns)
    correlation_matrix = correlation_matrix.corr()
    sns.heatmap(correlation_matrix, annot = True, cmap='Blues', vmin=0, vmax=1, annot_kws={"size": 6})
    plt.title(f'Correlation Heatmap for Features Across {year_1[:4]}-{year_3[-4:]} Seasons', pad = 20)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize = 6)
    plt.tight_layout()
    filename = 'visualizations/correlation_heatmap.png'
    plt.savefig(filename)
    plt.close()
    return filename

#Can open each type of visual by calling function and passing file path as argument
def open_scatterplot(file):
    scatterplot = mpimg.imread(file)
    plt.figure(figsize=(12, 6))
    plt.imshow(scatterplot)
    plt.axis('off')
    plt.show()

def open_histogram(file):
    histogram = mpimg.imread(file)
    plt.figure(figsize=(12, 6))
    plt.imshow(histogram)
    plt.axis('off')
    plt.show()

def open_correlation_heatmap(file):
    heatmap = mpimg.imread(file)
    plt.figure(figsize=(10,4 ))
    plt.imshow(heatmap)
    plt.axis('off')
    plt.tight_layout(pad=4)
    plt.show()

def open_barchart(file):
    barchart = mpimg.imread(file)
    plt.figure(figsize=(16, 6))
    plt.imshow(barchart)
    plt.axis('off')
    plt.show()