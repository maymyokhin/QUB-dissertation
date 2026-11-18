"""Melbourne_Dataset.ipynb
##Import Libraries##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from datetime import datetime
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import folium
from folium.plugins import HeatMap
import plotly.offline as pyo
import plotly.graph_objs as go
from wordcloud import WordCloud
from sklearn.linear_model import Lasso, LassoCV
import mgwr
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR
import libpysal as lp  # Required dependency for MGWR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.metrics import (r2_score, mean_absolute_error,
                            mean_squared_error, median_absolute_error,
                            explained_variance_score)
from scipy.stats import f_oneway
from sklearn.metrics import (r2_score, mean_absolute_error,
                           mean_squared_error, median_absolute_error,
                           explained_variance_score)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from statsmodels.api import OLS, add_constant
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error,
                            mean_squared_error, explained_variance_score)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
"""##Import##"""
pip install lime
import folium
from folium.plugins import HeatMap
import geopandas as gpd
from shapely.geometry import Point
from libpysal.weights import DistanceBand
from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import lime
import lime.lime_tabular
from datetime import datetime
"""##Load Dataset##"""
Melbourne = pd.read_csv("/content/MelbourneListings.csv")
"""##Initial Exploratory Data Analysis##"""
print(Melbourne.head())
print(Melbourne.info())
print(Melbourne['last_scraped'])
"""##Descriptive Statistics for numerical columns##"""
descriptive_stats=Melbourne.describe()
print(descriptive_stats)
# Get numerical columns (int and float)
num_cols = Melbourne.select_dtypes(include=['int64', 'float64']).columns
n_cols = len(num_cols)
# Calculate grid size dynamically
n_rows = (n_cols // 4) + (1 if n_cols % 4 != 0 else 0)  # 4 columns per row
plt.figure(figsize=(20, 5 * n_rows))  # Adjust height based on rows
for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, 4, i)
    sns.histplot(data=Melbourne, x=col, bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Visualization for categorical variables
# Identify categorical columns (excluding any list-type columns)
cat_cols = [col for col in Melbourne.select_dtypes(include=['object', 'category']).columns
            if not any(isinstance(x, list) for x in Melbourne[col].head())]

# Now create the visualization
plt.figure(figsize=(20, 10))
for i, col in enumerate(cat_cols[:8]):  # First 8 valid categorical columns
    plt.subplot(4, 2, i+1)

    # Convert to string if needed (safe handling)
    if Melbourne[col].apply(type).eq(list).any():
        plot_data = Melbourne[col].astype(str).value_counts().head(10)
    else:
        plot_data = Melbourne[col].value_counts().head(10)

    plot_data.plot(kind='bar')
    plt.title(f'Distribution of {col}', pad=10)
    plt.xticks(rotation=45)
    plt.xlabel('')

plt.tight_layout()
plt.show()

"""##Initial Data Exploration##"""

#Check for missing values
missing_values=Melbourne.isnull().sum()
print(missing_values)

#Check for duplicate rows
duplicate_rows= Melbourne.duplicated().sum()
print(duplicate_rows)

#Summary of data types
data_types= Melbourne.dtypes
print(data_types)

"""##Data Cleaning: Type Conversion and Initial Formatting##"""

# Convert price to numeric (remove $ and commas)
Melbourne['price'] = Melbourne['price'].str.replace('[$,]', '', regex=True).astype(float)
print("\n--- Price Column Head (After Cleaning) ---")
print(Melbourne['price'].head())

# Convert boolean columns
Melbourne['host_is_superhost'] = Melbourne['host_is_superhost'].map({'t': 1, 'f': 0})
Melbourne['instant_bookable'] = Melbourne['instant_bookable'].map({'t': 1, 'f': 0})

print("\n--- Superhost Status Descriptive Statistics ---")
print(Melbourne['host_is_superhost'].describe())
print("\n--- Instant Bookable Status Descriptive Statistics ---")
print(Melbourne['instant_bookable'].describe())

# Convert date columns
Melbourne['host_since'] = pd.to_datetime(Melbourne['host_since'])
Melbourne['last_review'] = pd.to_datetime(Melbourne['last_review'])
Melbourne['first_review'] = pd.to_datetime(Melbourne['first_review'])

# Extract year from host_since for analysis
Melbourne['year'] = Melbourne['host_since'].dt.year

"""Keep relevant variables"""

# List of relevant columns for price modeling
relevant_columns = [
    'id', 'name', 'host_id', 'host_name', 'host_since', 'host_is_superhost', 'neighbourhood', 'neighbourhood_cleansed', 'host_neighbourhood', 'host_total_listings_count', 'latitude', 'longitude', 'property_type',
    'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights', 'availability_365',
    'number_of_reviews', 'number_of_reviews_ltm', 'first_review', 'last_review', 'host_response_rate', 'host_acceptance_rate', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value', 'instant_bookable', 'calculated_host_listings_count', 'reviews_per_month', 'year'
]

# Filter the dataframe
Melbourne1 = Melbourne[relevant_columns].copy()

print("\n--- DataFrame Info After Selecting Relevant Columns ---")
Melbourne1.info()

batch_size = 12
num_cols = list(Melbourne1.select_dtypes(include=['int64', 'float64']).columns)

for batch in range(0, len(num_cols), batch_size):
    plt.figure(figsize=(20, 15))
    current_cols = num_cols[batch:batch+batch_size]

    for i, column in enumerate(current_cols, 1):
        plt.subplot(4, 3, i)
        sns.histplot(Melbourne1[column], kde=True, color='brown')
        plt.title(f'Distribution of {column}')

    plt.tight_layout()
    plt.show()

# Drop Price NA value
Melbourne1.dropna(subset=['price'], inplace=True)
print("\n--- Price Column Head (After Dropping NA) ---")
print(Melbourne1['price'].head())
print("\n--- Price Descriptive Statistics (After Cleaning) ---")
print(Melbourne1['price'].describe())

# Drop NaN values in host_since
print("\n--- Before Dropping host_since NA Values ---")
print(f"Total rows: {len(Melbourne1)}")
print(f"Missing host_since: {Melbourne1['host_since'].isna().sum()}")

Melbourne1.dropna(subset=['host_since'], inplace=True)

print("\n--- After Dropping host_since NA Values ---")
print(f"Remaining rows: {len(Melbourne1)}")
print(f"Earliest host since: {Melbourne1['host_since'].min()}")
print(f"Latest host since: {Melbourne1['host_since'].max()}")

"""##Data Cleaning##"""

# Calculate missing values
missing = Melbourne1.isnull().sum()
missing_percent = (missing / len(Melbourne1)) * 100

print("\n--- Missing Value Summary (Before Imputation) ---")
print(pd.DataFrame({'Missing Count': missing, 'Missing Percentage': missing_percent}).sort_values('Missing Count', ascending=False))

# Create a missing data matrix
plt.figure(figsize=(12, 8))
sns.heatmap(Melbourne1.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Matrix (Yellow = Missing)')
plt.xticks(rotation=90)
plt.show()

# Checking for outliers
num_cols = list(Melbourne1.select_dtypes(include=['int64', 'float64']).columns)

## Function to identify outliers using IQR method
def identify_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
# Identifying outliers for each numerical column
outliers_summary = {}
for col in num_cols:
    outliers = identify_outliers_iqr(Melbourne1, col)
    outliers_summary[col] = len(outliers)
outliers_summary
# Create boxplots for each numerical column
plt.figure(figsize=(15, 8))
sns.set(style="whitegrid")

# Calculate number of rows needed for subplots
n_cols = 7  # Number of columns in subplot grid
n_rows = (len(num_cols) + n_cols - 1) // n_cols  # Round up division

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
fig.suptitle('Boxplots of Numerical Columns with Outliers', y=1.02)

# Flatten axes array for easy iteration
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if i < len(axes):  # Ensure we don't exceed number of subplots
        sns.boxplot(x=Melbourne1[col], ax=axes[i], color='skyblue')
        axes[i].set_title(f'{col}\n(Outliers: {outliers_summary[col]})')
        axes[i].set_xlabel('')

# Hide any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

"""##Detailed Data Cleaning and Imputation##"""

cols_to_check = [
    'host_since', 'host_is_superhost', 'bathrooms',
    'bedrooms', 'beds', 'instant_bookable'
]

# Add review-related columns
review_cols = [col for col in Melbourne1.columns if col.startswith('review_')]
cols_to_check.extend(review_cols)

# Create a summary DataFrame
missing_analysis = pd.DataFrame({
    'total_rows': len(Melbourne1),
    'non_missing': Melbourne1[cols_to_check].count(),
    'missing': Melbourne1[cols_to_check].isna().sum(),
    'zero_values': (Melbourne1[cols_to_check] == 0).sum()
})

missing_analysis['missing_pct'] = (missing_analysis['missing'] / missing_analysis['total_rows']).round(3)
missing_analysis['zero_pct'] = (missing_analysis['zero_values'] / missing_analysis['total_rows']).round(3)

print(missing_analysis[['total_rows', 'non_missing', 'missing', 'missing_pct', 'zero_values', 'zero_pct']])

# ==============
# Detailed DATA CLEANING
# ===============
# Outlier Treatment
print("\n=== OUTLIER HANDLING ===")

# Price Outliers
price_upper_limit = Melbourne1['price'].quantile(0.99)
Melbourne1 = Melbourne1[Melbourne1['price'] <= price_upper_limit].copy()

# Visual Validation
plt.figure(figsize=(12, 5))
sns.boxplot(x=Melbourne1['price'])
plt.title('Price Distribution After Outlier Removal')
plt.show()

print("\nPrice Statistics After Cleaning:")
print(Melbourne1['price'].describe(percentiles=[.25, .5, .75, .9, .95, .99]))

# Other Numeric Features
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['beds', 'bedrooms', 'bathrooms']:
    Melbourne1 = remove_outliers(Melbourne1, col)

# Amenities Processing
print("\n=== AMENITIES PROCESSING ===")
Melbourne1['amenities'] = Melbourne1['amenities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Top Amenities Visualization
all_amenities = [amenity for sublist in Melbourne1['amenities'] for amenity in sublist]
amenity_counts = Counter(all_amenities)
amenity_df = pd.DataFrame.from_dict(amenity_counts, orient='index', columns=['count']).sort_values('count', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='count', y=amenity_df.index[:20], data=amenity_df[:20], palette='viridis')
plt.title('Top 20 Most Common Amenities')
plt.tight_layout()
plt.show()

# Missing Value Imputation
print("\n=== MISSING VALUE IMPUTATION ===")

# Categorical Columns
for col in ['host_neighbourhood', 'neighbourhood', 'host_name']:
    Melbourne1[col] = Melbourne1[col].fillna('unknown')

# Boolean Columns
Melbourne1['host_is_superhost'] = Melbourne1['host_is_superhost'].fillna(
    Melbourne1['host_is_superhost'].mode()[0]
)

# Numeric Columns
for col in ['bathrooms', 'bedrooms', 'beds']:
    Melbourne1[col] = Melbourne1[col].fillna(Melbourne1[col].mode()[0])

# Review Scores
review_columns = ['review_scores_rating', 'review_scores_accuracy',
                 'review_scores_cleanliness', 'review_scores_checkin',
                 'review_scores_communication', 'review_scores_location',
                 'review_scores_value']

for col in review_columns:
    Melbourne1[col] = Melbourne1[col].fillna(Melbourne1[col].median())

# Date Columns
for col in ['host_since', 'last_review', 'first_review']:
    Melbourne1[col] = Melbourne1[col].fillna(Melbourne1[col].median())

# Other Numeric Columns
for col in ['host_total_listings_count', 'reviews_per_month',
           'host_response_rate', 'host_acceptance_rate']:
    if Melbourne1[col].dtype == 'object':
        Melbourne1[col] = Melbourne1[col].astype(str).str.replace('%', '').astype(float) / 100
    Melbourne1[col] = Melbourne1[col].fillna(Melbourne1[col].mode()[0])

# Final Data Validation
print("\n=== FINAL DATA VALIDATION ===")

# Missing Values Check
print("\nRemaining Missing Values:")
print(Melbourne1.isnull().sum().sort_values(ascending=False))

# Host Analysis
print("\n--- Host Information ---")
print(f"Date Range: {Melbourne1['host_since'].min()} to {Melbourne1['host_since'].max()}")
print(f"Superhost Distribution:\n{Melbourne1['host_is_superhost'].value_counts()}")

# Neighborhood Analysis
plt.figure(figsize=(15, 7))
top_neighborhoods = Melbourne1['host_neighbourhood'].value_counts().index[:15]
sns.countplot(x="host_neighbourhood", data=Melbourne1, order=top_neighborhoods)
plt.title('Top 15 Neighborhoods by Listing Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Final Dataset Shape: {Melbourne1.shape}")
print("\n--- Final Descriptive Statistics ---")
print(Melbourne1.describe())

# Check for missing values in neighbourhood_cleansed column
missing_count = Melbourne1['neighbourhood_cleansed'].isnull().sum()
total_count = len(Melbourne1['neighbourhood_cleansed'])

print(f"Missing values in 'neighbourhood_cleansed': {missing_count}")
print(f"Total records: {total_count}")
print(f"Percentage missing: {missing_count/total_count*100:.2f}%")

# Visual representation
plt.figure(figsize=(8, 5))
sns.heatmap(Melbourne1[['neighbourhood_cleansed']].isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')
plt.title('Missing Values in neighbourhood_cleansed')
plt.show()

# Check for different types of missing/unknown values
na_count = Melbourne1['neighbourhood_cleansed'].isna().sum()
null_count = Melbourne1['neighbourhood_cleansed'].isnull().sum()
none_count = (Melbourne1['neighbourhood_cleansed'] == None).sum()
unknown_count = (Melbourne1['neighbourhood_cleansed'].str.lower() == 'unknown').sum()
blank_count = (Melbourne1['neighbourhood_cleansed'] == '').sum()

print("Missing/Unknown Value Analysis:")
print(f"- NA values: {na_count}")
print(f"- Null values: {null_count}")
print(f"- None values: {none_count}")
print(f"- 'Unknown' text values: {unknown_count}")
print(f"- Blank strings: {blank_count}")

# Combined missing/unknown percentage
total_missing = na_count + unknown_count + blank_count
total_records = len(Melbourne1)
print(f"\nTotal problematic values: {total_missing} ({total_missing/total_records*100:.2f}%)")

# Visual inspection
plt.figure(figsize=(10, 6))
missing_data = Melbourne1['neighbourhood_cleansed'].isna() | \
               (Melbourne1['neighbourhood_cleansed'].str.lower() == 'unknown') | \
               (Melbourne1['neighbourhood_cleansed'] == '')

sns.heatmap(missing_data.to_frame(),
            yticklabels=False,
            cbar=False,
            cmap='Reds')
plt.title('Missing/Unknown Values in Neighbourhood_Cleansed')
plt.show()

# Show sample of problematic rows
problem_rows = Melbourne1[missing_data]
if not problem_rows.empty:
    print("\nSample of problematic rows:")
    print(problem_rows[['neighbourhood_cleansed']].head())
else:
    print("\nNo missing or unknown values found!")

# Calculate summary statistics
min_nights_stats = Melbourne1['minimum_nights'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print("Minimum Nights Statistics:")
print(min_nights_stats)

max_nights_stats = Melbourne1['maximum_nights'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print("Maximum Nights Statistics:")
print(max_nights_stats)

plt.figure(figsize=(12, 6))

# Box plot
plt.subplot(1, 2, 1)
sns.boxplot(x=Melbourne1['minimum_nights'])
plt.title('Box Plot of Minimum Nights')

# Distribution plot (log scale often helps)
plt.subplot(1, 2, 2)
sns.histplot(Melbourne1['minimum_nights'], bins=50, log_scale=True)
plt.title('Distribution of Minimum Nights (Log Scale)')

plt.tight_layout()
plt.show()

# Calculate correlations
correlation_matrix = Melbourne1[['minimum_nights', 'maximum_nights', 'price']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize with heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Price vs. Min/Max Nights Correlation")
plt.show()

# Categorize minimum nights
bins = [0, 1, 3, 7, 30, 365, np.inf]
labels = ['1 night', '2-3 nights', '4-7 nights', '1-4 weeks', '1-12 months', '1yr+']
Melbourne1['min_stay_group'] = pd.cut(Melbourne1['minimum_nights'], bins=bins, labels=labels)

# Plot price distribution
plt.figure(figsize=(12, 6))
sns.boxplot(x='min_stay_group', y='price', data=Melbourne1, showfliers=False)
plt.title("Price Distribution by Minimum Stay Requirement")
plt.xticks(rotation=45)
plt.show()

# Categorize maximum nights
Melbourne1['max_stay_group'] = pd.cut(Melbourne1['maximum_nights'],
                                  bins=[0, 7, 30, 90, 365, np.inf],
                                  labels=['<1 week', '1-4 weeks', '1-3 months', '3-12 months', '1yr+'])

plt.figure(figsize=(12, 6))
sns.boxplot(x='max_stay_group', y='price', data=Melbourne1, showfliers=False)
plt.title("Price Distribution by Maximum Stay Allowed")
plt.xticks(rotation=45)
plt.show()

Melbourne1.describe()

"""##Visualizations Univariate Analysis##

"""

# Get numerical columns (int and float)
num_cols = Melbourne1.select_dtypes(include=['int64', 'float64']).columns
n_cols = len(num_cols)

# Calculate grid size dynamically
n_rows = (n_cols // 4) + (1 if n_cols % 4 != 0 else 0)  # 4 columns per row

plt.figure(figsize=(20, 5 * n_rows))  # Adjust height based on rows
for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, 4, i)
    sns.histplot(data=Melbourne1, x=col, bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Visualization for categorical variables
# Identify categorical columns (excluding any list-type columns)
cat_cols = [col for col in Melbourne1.select_dtypes(include=['object', 'category']).columns
            if not any(isinstance(x, list) for x in Melbourne1[col].head())]

# Now create the visualization
plt.figure(figsize=(16, 20))
for i, col in enumerate(cat_cols[:8]):  # First 8 valid categorical columns
    plt.subplot(4, 2, i+1)

    # Convert to string if needed (safe handling)
    if Melbourne1[col].apply(type).eq(list).any():
        plot_data = Melbourne1[col].astype(str).value_counts().head(10)
    else:
        plot_data = Melbourne1[col].value_counts().head(10)

    plot_data.plot(kind='bar')
    plt.title(f'Distribution of {col}', pad=10)
    plt.xticks(rotation=45)
    plt.xlabel('')

plt.tight_layout()
plt.show()

"""##Bivariate Analysis and Price Exploration##"""

"""Bivariate Analysis"""

# Calculate key percentiles for price
price_stats = Melbourne1['price'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print("\n--- Price Percentiles ---")
print(price_stats)

# Visualize price distribution with outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=Melbourne1['price'])
plt.title('Price Distribution with Outliers')
plt.show()

# Set style for better visuals
sns.set(style="whitegrid")

# Price Distribution by Room Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='room_type', y='price', data=Melbourne1)
plt.title('Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.show()

# Price Distribution by Room Type (Log Scale)
plt.figure(figsize=(12, 6))
sns.boxplot(x='room_type', y='price', data=Melbourne1)
plt.yscale('log')  # Log scale to better visualize wide price ranges
plt.title('Price Distribution by Room Type (Log Scale)')
plt.xlabel('Room Type')
plt.ylabel('Price ($) - Log Scale')
plt.xticks(rotation=45)
plt.show()

# Price Distribution by Top Property Types
# Get top N property types by count (adjust number as needed)
top_properties = Melbourne1['property_type'].value_counts().nlargest(10).index

plt.figure(figsize=(14, 6))
sns.boxplot(
    x='property_type',
    y='price',
    data=Melbourne1[Melbourne1['property_type'].isin(top_properties)],
    order=top_properties
)
plt.title('Price Distribution by Top Property Types')
plt.xlabel('Property Type')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.show()

# Price Distribution by Top Property Types (Log Scale)
plt.figure(figsize=(14, 6))
sns.boxplot(
    x='property_type',
    y='price',
    data=Melbourne1[Melbourne1['property_type'].isin(top_properties)],
    order=top_properties
)
plt.yscale('log')
plt.title('Price Distribution by Top Property Types (Log Scale)')
plt.xlabel('Property Type')
plt.ylabel('Price ($) - Log Scale')
plt.xticks(rotation=45)
plt.show()

# Average Price by Room and Property Type (Heatmap)
# Create a cross-tabulation of average prices
cross_tab = pd.pivot_table(
    Melbourne1[Melbourne1['property_type'].isin(top_properties)],
    values='price',
    index='property_type',
    columns='room_type',
    aggfunc=np.median
)

plt.figure(figsize=(12, 8))
sns.heatmap(cross_tab, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('Median Price by Property Type and Room Type ($)')
plt.xlabel('Room Type')
plt.ylabel('Property Type')
plt.show()

# Create the summary table
price_summary = Melbourne1.groupby(['year', 'room_type']).agg(
    Listings_Count=('id', 'count'),
    Avg_Price=('price', 'mean'),
    Median_Price=('price', 'median')
).reset_index()

# Pivot to get room types as columns
pivoted_price = price_summary.pivot_table(
    index='year',
    columns='room_type',
    values=['Avg_Price', 'Median_Price'],
    fill_value=0
)

# Flatten multi-index columns
pivoted_price.columns = [f"{col[1]}_{col[0]}" for col in pivoted_price.columns]
pivoted_price.reset_index(inplace=True)

# Format the table for display
formatted_price_table = pivoted_price.copy()
formatted_price_table['Year'] = formatted_price_table['year']
formatted_price_table = formatted_price_table[[
    'Year',
    'Entire home/apt_Avg_Price', 'Private room_Avg_Price', 'Shared room_Avg_Price',
    'Entire home/apt_Median_Price', 'Private room_Median_Price', 'Shared room_Median_Price'
]]

# Rename columns to match your requested format
formatted_price_table.columns = [
    'Year',
    'Avg Price (Entire Home)', 'Avg Price (Private Room)', 'Avg Price (Shared Room)',
    'Median Price (Entire Home)', 'Median Price (Private Room)', 'Median Price (Shared Room)'
]

# Apply currency formatting
for col in formatted_price_table.columns[1:]:
    formatted_price_table[col] = formatted_price_table[col].apply(lambda x: f"${x:,.2f}")

print("\n--- Formatted Price Table by Year and Room Type (Avg/Median) ---")
display(formatted_price_table)

# Create summary statistics
price_stats_detailed = Melbourne1.groupby(['year', 'room_type'])['price'].agg(
    Listings_Count='count',
    Min_Price='min',
    Max_Price='max',
    Avg_Price='mean',
    Median_Price='median'
).reset_index()

# Pivot to get room types as columns
pivoted_stats = price_stats_detailed.pivot_table(
    index='year',
    columns='room_type',
    values=['Min_Price', 'Max_Price', 'Avg_Price', 'Median_Price'],
    fill_value=0
)

# Flatten multi-index columns
pivoted_stats.columns = [f"{col[1]}_{col[0]}" for col in pivoted_stats.columns]
pivoted_stats.reset_index(inplace=True)

# Create final formatted table
final_price_table_detailed = pivoted_stats[[
    'year',
    'Entire home/apt_Min_Price', 'Private room_Min_Price', 'Shared room_Min_Price',
    'Entire home/apt_Max_Price', 'Private room_Max_Price', 'Shared room_Max_Price',
    'Entire home/apt_Avg_Price', 'Private room_Avg_Price', 'Shared room_Avg_Price',
    'Entire home/apt_Median_Price', 'Private room_Median_Price', 'Shared room_Median_Price'
]]

# Rename columns
final_price_table_detailed.columns = [
    'Year',
    'Min (Entire)', 'Min (Private)', 'Min (Shared)',
    'Max (Entire)', 'Max (Private)', 'Max (Shared)',
    'Avg (Entire)', 'Avg (Private)', 'Avg (Shared)',
    'Median (Entire)', 'Median (Private)', 'Median (Shared)'
]

# Apply currency formatting
price_cols = final_price_table_detailed.columns[1:]
final_price_table_detailed[price_cols] = final_price_table_detailed[price_cols].applymap(lambda x: f"${x:,.2f}")

print("\n--- Detailed Price Table by Year and Room Type (Min/Max/Avg/Median) ---")
display(final_price_table_detailed)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Beds
sns.histplot(Melbourne1['beds'].dropna(), bins=30, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Beds')
axes[0].set_xlabel('Number of Beds')
axes[0].set_ylabel('Count')

# Bedrooms
sns.histplot(Melbourne1['bedrooms'].dropna(), bins=30, ax=axes[1], color='salmon')
axes[1].set_title('Distribution of Bedrooms')
axes[1].set_xlabel('Number of Bedrooms')

# Bathrooms
sns.histplot(Melbourne1['bathrooms'].dropna(), bins=30, ax=axes[2], color='lightgreen')
axes[2].set_title('Distribution of Bathrooms')
axes[2].set_xlabel('Number of Bathrooms')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=Melbourne1[['beds', 'bedrooms', 'bathrooms']], palette="Set2")
plt.title('Beds, Bedrooms, and Bathrooms Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Numeric features vs price
num_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'number_of_reviews', 'review_scores_rating']

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=Melbourne1[col], y=Melbourne1['price'], alpha=0.5)
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()

# Correlation Matrix
print("\n--- Correlation Matrix of Numerical Features ---")
numerical_cols = Melbourne1.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = Melbourne1[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix (Numerical Features)')
plt.show()

price_corr = corr_matrix['price'].sort_values(ascending=False)
print("\n--- Top features correlated with price ---")
print(price_corr)

# Price Distribution (Original vs Log Scale)
plt.figure(figsize=(12, 6))
sns.histplot(Melbourne1['price'], bins=50, kde=True)
plt.title('Price Distribution (Original Scale)')
plt.xlabel('Price ($)')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(Melbourne1['price']), bins=50, kde=True)
plt.title('Price Distribution (Log Scale)')
plt.xlabel('log(Price)')
plt.show()

# Price by category (room_type)
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='room_type',
    y='price',
    data=Melbourne1
)
plt.title('Price Distribution by Room Type')
plt.xticks(rotation=45)
plt.show()

# Price by category (property_type)
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='property_type',
    y='price',
    data=Melbourne1
)
plt.title('Price Distribution by Property Type')
plt.xticks(rotation=45)
plt.show()

# Price by category (host_is_superhost)
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='host_is_superhost',
    y='price',
    data=Melbourne1
)
plt.title('Price Distribution by host_is_superhost')
plt.xticks(rotation=45)
plt.show()

# Price by category (host_neighbourhood) - showing top 10 neighbourhoods by listing count
top_neighbourhoods = Melbourne1['host_neighbourhood'].value_counts().nlargest(10).index
plt.figure(figsize=(15, 7))
sns.boxplot(
    x='host_neighbourhood',
    y='price',
    data=Melbourne1[Melbourne1['host_neighbourhood'].isin(top_neighbourhoods)],
    order=top_neighbourhoods
)
plt.title('Price Distribution by Top 10 Neighbourhoods')
plt.xticks(rotation=90)
plt.show()

"""##Data Quality Checks and Further Refinements ##"""

# Host Since Analysis
print("\n=== Host Since ===")
print("Date Range:",
      Melbourne1['host_since'].min().strftime('%Y-%m-%d'), "to",
      Melbourne1['host_since'].max().strftime('%Y-%m-%d'))
print("Median Host Tenure:",
      (datetime.now() - Melbourne1['host_since'].median()).days, "days")

# Last Review Analysis (with missing value handling)
print("\n=== Last Review ===")
if Melbourne1['last_review'].notna().any():
    print("Date Range (non-null):",
          Melbourne1['last_review'].min().strftime('%Y-%m-%d'), "to",
          Melbourne1['last_review'].max().strftime('%Y-%m-%d'))
else:
    print("No valid review dates found")

print("Null Values:", Melbourne1['last_review'].isna().sum())

# Review Date Consistency Check
active_without_reviews = Melbourne1[
    (Melbourne1['number_of_reviews'] > 0) &
    (Melbourne1['last_review'].isna())
]

print("\n=== Data Quality Issues (Date) ===")
print(f"Listings with reviews but missing 'last_review' dates: {len(active_without_reviews)}")
print(f"Listings with future 'last_review' dates: {len(Melbourne1[Melbourne1['last_review'] > datetime.now()])}")

# Host Response Rate and Acceptance Rate Analysis
print("\n--- Host Response Rate Info ---")
print(Melbourne1['host_response_rate'].info())
print("\n--- Host Response Rate Head ---")
print(Melbourne1['host_response_rate'].head())
print("\n--- Host Response Rate Description ---")
print(Melbourne1['host_response_rate'].describe())

print("\n--- Host Acceptance Rate Info ---")
print(Melbourne1['host_acceptance_rate'].info())
print("\n--- Host Acceptance Rate Head ---")
print(Melbourne1['host_acceptance_rate'].head())
print("\n--- Host Acceptance Rate Info (re-check) ---")
print(Melbourne1['host_acceptance_rate'].info())

# Convert percentage strings to numeric (re-run in case previous imputation didn't handle it fully)
def convert_percentage(rate):
    if pd.isna(rate):
        return np.nan
    # Check if it's already a float (from previous imputation)
    if isinstance(rate, (float, int)):
        return float(rate)
    return float(str(rate).strip('%'))/100

Melbourne1['host_response_rate_num'] = Melbourne1['host_response_rate'].apply(convert_percentage)
Melbourne1['host_acceptance_rate_num'] = Melbourne1['host_acceptance_rate'].apply(convert_percentage)

# Check missingness correlation
missing_data_rates = Melbourne1[['host_response_rate_num', 'host_acceptance_rate_num']].isnull()
print("\n--- Missing Value Correlation (Response/Acceptance Rates) ---")
print(missing_data_rates.corr())

# Compare superhost vs non-superhost
print("\n--- Missing rates by superhost status ---")
print(missing_data_rates.groupby(Melbourne1['host_is_superhost']).mean())

# Comparative analysis with review scores
review_cols = ['review_scores_rating', 'review_scores_communication']
for rate in ['host_response_rate_num', 'host_acceptance_rate_num']:
    print(f"\n--- Correlation between {rate} and review scores ---")
    print(Melbourne1[[rate] + review_cols].corr())

performance_metrics = ['number_of_reviews', 'price', 'availability_365']
for rate in ['host_response_rate_num', 'host_acceptance_rate_num']:
    plt.figure(figsize=(10,4))
    for i, metric in enumerate(performance_metrics):
        plt.subplot(1,3,i+1)
        sns.scatterplot(x=rate, y=metric, data=Melbourne1, alpha=0.3)
        plt.title(f'{rate} vs {metric}')
    plt.tight_layout()
    plt.show()

# Value range validation
# Check for impossible values
issues = {
    'Negative Prices': len(Melbourne1[Melbourne1['price'] <= 0]),
    'Zero Accommodates': len(Melbourne1[Melbourne1['accommodates'] == 0]),
    'Availability >365': len(Melbourne1[Melbourne1['availability_365'] > 365]),
    'Bedrooms >20': len(Melbourne1[Melbourne1['bedrooms'] > 20]),
    'Bathrooms >10': len(Melbourne1[Melbourne1['bathrooms'] > 10])
}
print("\n--- Data Quality Issues (Value Ranges) ---")
for issue, count in issues.items():
    print(f"{issue}: {count}")

# Get listings with more than 10 bathrooms
luxury_bathrooms = Melbourne1[Melbourne1['bathrooms'] > 10]
print("\n--- Listings with more than 10 bathrooms ---")
display(luxury_bathrooms[[
    'id', 'name', 'host_name', 'property_type',
    'bathrooms', 'bedrooms', 'accommodates', 'price',
    'latitude', 'longitude'
]].sort_values('bathrooms', ascending=False))

# Review score validation
# All review scores should be between 1-10
review_cols_num = [c for c in Melbourne1.columns if c.startswith('review_scores_') and c != 'reviews_per_month'] # Exclude reviews_per_month
invalid_reviews = Melbourne1[(Melbourne1[review_cols_num] < 1).any(axis=1) |
                             (Melbourne1[review_cols_num] > 10).any(axis=1)]
print("\n--- Entries with invalid review scores (outside 1-10 range) ---")
print(f"Number of listings with invalid review scores: {len(invalid_reviews)}")

print("\nReview Score Columns Checked:", review_cols_num)

# Logical relationship (Capacity Consistency)
capacity_issues_df = Melbourne1[
    (Melbourne1['beds'] > Melbourne1['accommodates']) |
    (Melbourne1['bedrooms'] > Melbourne1['accommodates']) |
    (Melbourne1['bathrooms'] > Melbourne1['accommodates'])
]
print("\n--- Properties with initial capacity inconsistencies ---")
print(f"Total properties with capacity inconsistencies: {len(capacity_issues_df)}")

# Breakdown of inconsistency types
print(f"Beds > Accommodates: {len(capacity_issues_df[capacity_issues_df['beds'] > capacity_issues_df['accommodates']])}")
print(f"Bedrooms > Accommodates: {len(capacity_issues_df[capacity_issues_df['bedrooms'] > capacity_issues_df['accommodates']])}")
print(f"Bathrooms > Accommodates: {len(capacity_issues_df[capacity_issues_df['bathrooms'] > capacity_issues_df['accommodates']])}")

def correct_capacity(df):
    """Corrects capacity inconsistencies where beds/bedrooms/bathrooms exceed accommodates."""
    original = df.copy()

    # Apply corrections for each column
    for col in ['beds', 'bedrooms', 'bathrooms']:
        # Only correct if the value is greater than accommodates
        mask = (df[col].notna()) & (df['accommodates'].notna()) & (df[col] > df['accommodates'])
        df.loc[mask, col] = df.loc[mask, 'accommodates']

    # Report changes
    print("\n--- Capacity Correction Report ---")
    for col in ['beds', 'bedrooms', 'bathrooms']:
        changed = (original[col] != df[col]).sum()
        print(f"Corrected {changed} {col} values.")

    return df

Melbourne1 = correct_capacity(Melbourne1)

def validate_capacity(df):
    """Validates if capacity issues remain after correction."""
    issues = {
        'beds': len(df[df['beds'] > df['accommodates']]),
        'bedrooms': len(df[df['bedrooms'] > df['accommodates']]),
        'bathrooms': len(df[df['bathrooms'] > df['accommodates']])
    }
    if any(issues.values()):
        print("\n--- Remaining issues after correction ---")
        for k, v in issues.items():
            print(f"{k}: {v}")
    else:
        print("\n--- All capacity constraints validated successfully! ---")

validate_capacity(Melbourne1)

"""##Initial Data Exploration and Preprocessing##"""

print("\n--- Host Neighbourhood Information ---")
print(Melbourne1['host_neighbourhood'].info())
print(Melbourne1['host_neighbourhood'].head())
print(Melbourne1['host_neighbourhood'].describe())

print("\n--- Neighbourhood Information ---")
print(Melbourne1['neighbourhood_cleansed'].info())
print(Melbourne1['neighbourhood_cleansed'].head())
print(Melbourne1['neighbourhood_cleansed'].describe())

# Check missingness patterns
print("\n--- Missing Value Correlation Matrix ---")
missing_corr = Melbourne1.isnull().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(missing_corr, cmap='coolwarm', center=0)
plt.title('Missing Value Correlation Matrix')
plt.show()

# Price comparison for top 5 neighborhoods vs Unknown
plt.figure(figsize=(10, 6))
sns.boxplot(x='host_neighbourhood', y='price',
            data=Melbourne1[Melbourne1['host_neighbourhood'].isin(
                ['Unknown'] + list(Melbourne1['host_neighbourhood'].value_counts().index[:5]))])
plt.xticks(rotation=45)
plt.title('Price Distribution by Host Neighborhood (Top 5 vs Unknown)')
plt.show()

# Superhost rate comparison
print("\n--- Superhost Rates by Host Neighbourhood ---")
superhost_rates = Melbourne1.groupby('host_neighbourhood')['host_is_superhost'].mean().sort_values(ascending=False)
print(superhost_rates.head(10))

# T-test for price differences between known and unknown host neighborhoods
known_price = Melbourne1[Melbourne1['host_neighbourhood'] != 'Unknown']['price']
unknown_price = Melbourne1[Melbourne1['host_neighbourhood'] == 'Unknown']['price']

t_stat, p_value = ttest_ind(known_price, unknown_price, equal_var=False)
print(f"\n--- T-test p-value for Price (Known vs Unknown Host Neighbourhood): {p_value:.4f} ---")

# Cross-tab with property type
print("\n--- Cross-tab: Host Neighbourhood vs Property Type ---")
print(pd.crosstab(Melbourne1['host_neighbourhood'],
                  Melbourne1['property_type'],
                  normalize='index').style.background_gradient(cmap='Blues'))

# Geographic plot of known vs unknown
known = Melbourne1[Melbourne1['host_neighbourhood'] != 'Unknown']
unknown = Melbourne1[Melbourne1['host_neighbourhood'] == 'Unknown']

plt.figure(figsize=(12, 8))
plt.scatter(known['longitude'], known['latitude'], alpha=0.3, label='Known')
plt.scatter(unknown['longitude'], unknown['latitude'], alpha=0.3, c='red', label='Unknown')
plt.legend()
plt.title('Geographic Distribution of Missing Host Neighborhoods')
plt.show()

# Temporal analysis (if host_since available)
Melbourne1['host_since'] = pd.to_datetime(Melbourne1['host_since'])
Melbourne1['host_tenure'] = (pd.to_datetime('today') - Melbourne1['host_since']).dt.days
sns.kdeplot(data=Melbourne1, x='host_tenure', hue='host_neighbourhood', common_norm=False)
plt.title('Host Tenure Distribution by Neighborhood Status')
plt.show()

# Missingness analysis for neighbourhood_cleansed
print("\n=== Missing Value Analysis for neighbourhood_cleansed ===")

# 1. Basic missing value stats
missing_count = Melbourne1['neighbourhood_cleansed'].isna().sum()
unknown_count = (Melbourne1['neighbourhood_cleansed'] == 'Unknown').sum()
total_records = len(Melbourne1)
print(f"Missing values: {missing_count} ({missing_count/total_records*100:.2f}%)")
print(f"'Unknown' values: {unknown_count} ({unknown_count/total_records*100:.2f}%)")

# 2. Price comparison for top 5 neighborhoods vs Missing/Unknown
plt.figure(figsize=(12, 6))
top_neighbourhoods = Melbourne1['neighbourhood_cleansed'].value_counts().index[:5]

# Create comparison groups
comparison_groups = list(top_neighbourhoods) + ['Unknown']
if missing_count > 0:
    comparison_groups.append('Missing')

# Prepare data for visualization
plot_data = Melbourne1.copy()
plot_data['neighbourhood_group'] = np.where(
    plot_data['neighbourhood_cleansed'].isna(),
    'Missing',
    plot_data['neighbourhood_cleansed']
)
plot_data = plot_data[plot_data['neighbourhood_group'].isin(comparison_groups)]

sns.boxplot(
    x='neighbourhood_group',
    y='price',
    data=plot_data,
    order=comparison_groups
)
plt.xticks(rotation=45)
plt.title('Price Distribution by Neighbourhood Status (Top 5 vs Unknown/Missing)')
plt.show()

# 3. Superhost rate comparison
print("\n--- Superhost Rates by Neighbourhood ---")
neighbourhood_stats = Melbourne1.groupby('neighbourhood_cleansed').agg(
    superhost_rate=('host_is_superhost', 'mean'),
    avg_price=('price', 'mean'),
    count=('id', 'count')
).sort_values('count', ascending=False)
print(neighbourhood_stats.head(10))

# 4. T-test for price differences
known_price = Melbourne1[~Melbourne1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]['price']
unknown_price = Melbourne1[Melbourne1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]['price']

t_stat, p_value = ttest_ind(known_price, unknown_price, equal_var=False)
print(f"\nT-test p-value for Price (Known vs Unknown/Missing): {p_value:.4f}")

# 5. Cross-tab with property type
print("\n--- Property Type Distribution ---")
print(pd.crosstab(
    Melbourne1['neighbourhood_cleansed'],
    Melbourne1['property_type'],
    margins=True
).style.background_gradient(cmap='Blues'))

# 6. Geographic plot
plt.figure(figsize=(12, 8))
known = Melbourne1[~Melbourne1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]
unknown = Melbourne1[Melbourne1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]

plt.scatter(known['longitude'], known['latitude'], alpha=0.3, label='Known')
plt.scatter(unknown['longitude'], unknown['latitude'], alpha=0.5, c='red', label='Unknown/Missing')
plt.legend()
plt.title('Geographic Distribution of Neighbourhood Status')
plt.show()

# 7. Temporal analysis (only if host_since exists)
if 'host_since' in Melbourne1.columns:
    # Create host_tenure column if it doesn't exist
    if 'host_tenure' not in Melbourne1.columns:
        Melbourne1['host_tenure'] = (pd.to_datetime('today') - pd.to_datetime(Melbourne1['host_since'])).dt.days

    plt.figure(figsize=(10, 6))
    # Create simplified groups for visualization
    plot_data = Melbourne1.copy()
    plot_data['neighbourhood_status'] = np.where(
        plot_data['neighbourhood_cleansed'].isna(),
        'Missing',
        np.where(
            plot_data['neighbourhood_cleansed'] == 'Unknown',
            'Unknown',
            'Known'
        )
    )

    sns.kdeplot(
        data=plot_data,
        x='host_tenure',
        hue='neighbourhood_status',
        hue_order=['Known', 'Unknown', 'Missing'],
        common_norm=False
    )
    plt.title('Host Tenure by Neighbourhood Status')
    plt.xlabel('Host Tenure (days)')
    plt.show()
else:
    print("\nNote: 'host_since' column not available for temporal analysis")

Melbourne1.to_csv('melbourne_airbnb_listings.csv', index=False)
print("Saved as CSV: melbourne_airbnb_listings.csv")

"""##VIF##

VIF Variance Inflation Factor for Multicollinearity
"""

#Vif Check

# Assume your DataFrame is called df
# Step 1: Drop irrelevant columns
exclude_cols = [
    'id', 'name', 'host_id', 'host_name', 'host_since', 'neighbourhood', 'host_neighbourhood', 'neighbourhood_cleansed',
    'property_type', 'room_type', 'amenities', 'last_review', 'host_since', 'first_review' # categorical or object columns
]

# Also drop boolean and datetime columns for VIF calculation
exclude_types = ['bool', 'datetime64[ns]', 'category', 'object']

# Filter to only numeric columns not in exclude_cols
vif_Sydney = Melbourne1.drop(columns=exclude_cols)
vif_Sydney = vif_Sydney.select_dtypes(exclude=exclude_types)

# Step 2: Add constant for VIF calculation
X = add_constant(vif_Sydney)

# Step 3: Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Step 4: Sort and display
print(vif_data.sort_values(by="VIF", ascending=False))

"""##Correlation##"""

# Calculate Spearman correlation matrix
spearman_corr = Melbourne1.select_dtypes(include=['int64','float64']).corr(method='spearman')

# Plot heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Spearman Correlation Matrix')
plt.show()

# Get top features correlated with price
price_corr_spearman = spearman_corr['price'].sort_values(key=abs, ascending=False)
print("\nTop features correlated with price (Spearman):")
print(price_corr_spearman.head(15))

"""##Feature selection##"""

from sklearn.ensemble import RandomForestRegressor

# Prepare data
X = Melbourne1.select_dtypes(include=['int64','float64']).drop(columns=['price'])
y = Melbourne1['price']

# Fit model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Random Forest Feature Importance')
plt.show()

import xgboost as xgb

# Fit model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X, y)

# Get feature importance
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importance.head(15))
plt.title('XGBoost Feature Importance')
plt.show()

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Lasso with cross-validation to find best alpha
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# Get coefficients
lasso_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=lasso_coef.head(15))
plt.title('Lasso Regression Feature Importance')
plt.show()

# Features kept by Lasso (non-zero coefficients)
selected_features = X.columns[lasso.coef_ != 0]
print("\nFeatures selected by Lasso:")
print(selected_features)

# Combine results from all methods
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Spearman': abs(spearman_corr['price'][X.columns]),
    'RF': rf.feature_importances_,
    'XGB': xgb_model.feature_importances_,
    'Lasso': abs(lasso.coef_)
})

# Normalize scores
feature_scores['RF'] = feature_scores['RF'] / feature_scores['RF'].max()
feature_scores['XGB'] = feature_scores['XGB'] / feature_scores['XGB'].max()
feature_scores['Lasso'] = feature_scores['Lasso'] / feature_scores['Lasso'].max()

# Calculate consensus score
feature_scores['Consensus'] = feature_scores[['Spearman', 'RF', 'XGB', 'Lasso']].mean(axis=1)

# Get top features
top_features = feature_scores.sort_values('Consensus', ascending=False).head(15)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Consensus', y='Feature', data=top_features)
plt.title('Consensus Feature Importance')
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (r2_score, mean_squared_error,
                           mean_absolute_error, mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.inspection import PartialDependenceDisplay

# ======================
# FEATURE SELECTION
# ======================
selected_features = [
    'bedrooms', 'longitude', 'latitude', 'bathrooms',
    'accommodates', 'review_scores_location', 'beds',
    'availability_365', 'host_total_listings_count',
    'reviews_per_month'
]

# Define a function to calculate all metrics
def calculate_metrics(model, X_train, y_train, X_test, y_test, cv=None):
    """Calculate all requested metrics for train and test sets"""
    # Train metrics
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'Train R2': r2_score(y_train, y_train_pred),
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Train MAPE': mean_absolute_percentage_error(y_train, y_train_pred)
    }

    # Test metrics
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'Test R2': r2_score(y_test, y_test_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred)
    }

    # Cross-validation metrics if cv is provided
    cv_metrics = {}
    if cv:
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        cv_metrics = {
            'CV R2 Mean': np.mean(cv_r2),
            'CV R2 Std': np.std(cv_r2),
            'CV R2 Min': np.min(cv_r2),
            'CV R2 Max': np.max(cv_r2)
        }

    return {**train_metrics, **test_metrics, **cv_metrics}

# Prepare data
X = Melbourne1[selected_features]
y = Melbourne1['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize models (same as before)
models = {
    'OLS': LinearRegression(),
    'Lasso': Lasso(alpha=0.01),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'XGBoost': XGBRegressor(random_state=42)
}

# Set up cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate all models
results = []
for name, model in models.items():
    # Handle scaling for models that need it
    if name in ['OLS', 'Lasso', 'SVR', 'KNN']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # Fit model
    model.fit(X_tr, y_train)

    # Calculate metrics
    metrics = calculate_metrics(model, X_tr, y_train, X_te, y_test, cv=cv)

    # Store results
    results.append({
        'Model': name,
        **metrics
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# Reorder columns for better readability
column_order = [
    'Model',
    'Train R2', 'Test R2', 'CV R2 Mean', 'CV R2 Std',
    'Train MSE', 'Test MSE',
    'Train RMSE', 'Test RMSE',
    'Train MAE', 'Test MAE',
    'Train MAPE', 'Test MAPE',
    'CV R2 Min', 'CV R2 Max'
]
results_df = results_df[column_order]

# Display results sorted by Test R2
print(results_df.sort_values('Test R2', ascending=False).round(4))

# Save to CSV if needed
results_df.to_csv('model_performance_metrics.csv', index=False)

import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay

# Prepare data (scaled for KNN/OLS)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# 1. Gradient Boosting - In-Depth Analysis
# =============================================
gb = GradientBoostingRegressor(random_state=42)

# Parameter tuning
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

gb_cv = RandomizedSearchCV(gb, gb_param_grid, n_iter=20, cv=5,
                          scoring='r2', random_state=42)
gb_cv.fit(X_train, y_train)

print("=== Gradient Boosting ===")
print("Best Parameters:", gb_cv.best_params_)
print(f"Best CV R2: {gb_cv.best_score_:.4f}")

# Feature Importance
gb_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': gb_cv.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=gb_importance.head(10))
plt.title("Gradient Boosting Feature Importance (Top 10)")
plt.show()

# Partial Dependence Plots
print("\nPartial Dependence Plots (Top 3 Features):")
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    gb_cv.best_estimator_,
    X_train,
    features=[0, 1, 2],
    feature_names=selected_features,
    ax=ax
)
plt.suptitle("Gradient Boosting Partial Dependence")
plt.tight_layout()
plt.show()

# =============================================
# 2. KNN - In-Depth Analysis
# =============================================
knn = KNeighborsRegressor()

# Parameter tuning
knn_param_grid = {
    'n_neighbors': range(3, 21),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1: manhattan, 2: euclidean
}

knn_cv = GridSearchCV(knn, knn_param_grid, cv=5, scoring='r2')
knn_cv.fit(X_train_scaled, y_train)

print("\n=== KNN ===")
print("Best Parameters:", knn_cv.best_params_)
print(f"Best CV R2: {knn_cv.best_score_:.4f}")

# Distance Analysis
distances, indices = knn_cv.best_estimator_.kneighbors(X_train_scaled[:5])  # First 5 samples
print("\nNeighbor distances for first 5 samples:")
print(distances)

# =============================================
# 3. OLS - In-Depth Analysis with Statsmodels
# =============================================
# Add constant for statsmodels (intercept term)
X_train_const = sm.add_constant(X_train_scaled)
X_test_const = sm.add_constant(X_test_scaled)

# Fit OLS model
ols = sm.OLS(y_train, X_train_const)
ols_results = ols.fit()

print("\n=== OLS Regression Summary ===")
print(ols_results.summary())

# Create coefficient plot
coef_df = pd.DataFrame({
    'Feature': ['Intercept'] + selected_features,
    'Coefficient': ols_results.params,
    'P-value': ols_results.pvalues
}).sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(15))
plt.title("OLS Coefficient Magnitudes (Top 15)")
plt.axvline(0, color='k', linestyle='--')
plt.show()

# Random Forest analysis
rf = RandomForestRegressor(random_state=42, oob_score=True)
rf.fit(X_train, y_train)

# OOB score
print("OOB R2:", rf.oob_score_)

# Tune parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_depth': [5, 10, 20, None]
}

rf_cv = RandomizedSearchCV(rf, param_grid, n_iter=20, cv=5, scoring='r2', random_state=42)
rf_cv.fit(X_train, y_train)

print("Best Random Forest Parameters:", rf_cv.best_params_)
print("Best CV R2:", rf_cv.best_score_)

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_cv.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=rf_importance)
plt.title("Random Forest Feature Importance")
plt.show()

# XGBoost analysis
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)

# Tune parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_cv = RandomizedSearchCV(xgb, param_grid, n_iter=20, cv=5, scoring='r2', random_state=42)
xgb_cv.fit(X_train, y_train)

print("Best XGBoost Parameters:", xgb_cv.best_params_)
print("Best CV R2:", xgb_cv.best_score_)

# Feature importance
xgb_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': xgb_cv.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance)
plt.title("XGBoost Feature Importance")
plt.show()

# Partial dependence plots
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    xgb_cv.best_estimator_,
    X_train,
    features=[0, 1, 2],  # Top 3 features
    feature_names=selected_features,
    ax=ax
)
plt.suptitle("Partial Dependence Plots")
plt.tight_layout()
plt.show()

from sklearn.metrics import (r2_score, mean_squared_error,
                           mean_absolute_error, mean_absolute_percentage_error)
from sklearn.model_selection import cross_val_score, KFold

# Initialize models with their best parameters from tuning
tuned_models = {
    'Random Forest': rf_cv.best_estimator_,
    'XGBoost': xgb_cv.best_estimator_,
    'Gradient Boosting': gb_cv.best_estimator_,
    'KNN': knn_cv.best_estimator_,
    'OLS': LinearRegression()  # Using sklearn's implementation for consistency
}

# Set up evaluation framework
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a model and return comprehensive metrics"""
    # Handle scaling for models that need it
    if model_name in ['KNN', 'OLS']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # Fit model
    model.fit(X_tr, y_train)

    # Predictions
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train R2': r2_score(y_train, y_train_pred),
        'Test R2': r2_score(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train MAPE': mean_absolute_percentage_error(y_train, y_train_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred)
    }

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='r2')
    metrics.update({
        'CV R2 Mean': np.mean(cv_scores),
        'CV R2 Std': np.std(cv_scores)
    })

    return metrics

# Evaluate all models
results = []
for name, model in tuned_models.items():
    results.append(evaluate_model(model, X_train, y_train, X_test, y_test, name))

# Create results dataframe
results_df = pd.DataFrame(results)

# Reorder columns for better readability
column_order = [
    'Model',
    'Train R2', 'Test R2', 'CV R2 Mean', 'CV R2 Std',
    'Train RMSE', 'Test RMSE',
    'Train MAE', 'Test MAE',
    'Train MAPE', 'Test MAPE'
]
results_df = results_df[column_order]

# Display results sorted by Test R2
print("=== Model Performance Comparison ===")
print(results_df.sort_values('Test R2', ascending=False).round(4))

# =============================================
# Visualization 1: Model Comparison Bar Plot
# =============================================
plt.figure(figsize=(14, 8))
metrics_to_plot = ['Test R2', 'Test RMSE', 'Test MAE']

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(1, 3, i)
    sns.barplot(x=metric, y='Model', data=results_df.sort_values(metric, ascending=i>1))
    plt.title(metric)
    if metric == 'Test R2':
        plt.xlim(0, 1)  # R2 ranges 0-1
plt.tight_layout()
plt.suptitle("Model Performance Comparison (Test Set Metrics)", y=1.02)
plt.show()

# =============================================
# Visualization 2: Actual vs Predicted Values
# =============================================
plt.figure(figsize=(15, 10))
for i, (name, model) in enumerate(tuned_models.items(), 1):
    # Handle scaling
    if name in ['KNN', 'OLS']:
        X_te = X_test_scaled
    else:
        X_te = X_test

    y_pred = model.predict(X_te)

    plt.subplot(2, 3, i)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name}\nTest R2: {results_df[results_df['Model']==name]['Test R2'].values[0]:.3f}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    # Add histogram of residuals
    if i == len(tuned_models):  # Last subplot
        plt.subplot(2, 3, i+1)
        residuals = np.concatenate([y_test - model.predict(X_te)
                                  for model in tuned_models.values()])
        model_names = np.repeat(list(tuned_models.keys()), len(y_test))
        sns.boxplot(x=residuals, y=model_names)
        plt.title("Residual Distribution Comparison")
        plt.xlabel("Residuals")
        plt.ylabel("Model")
plt.tight_layout()
plt.show()

# =============================================
# Visualization 3: Metric Comparison Radar Chart
# =============================================
from math import pi

# Normalize metrics for radar chart (0-1 scale)
def normalize_series(s):
    if s.name in ['Test R2', 'CV R2 Mean']:  # Higher is better
        return (s - s.min()) / (s.max() - s.min())
    else:  # RMSE, MAE, MAPE - lower is better
        return 1 - ((s - s.min()) / (s.max() - s.min()))

radar_df = results_df.set_index('Model')[['Test R2', 'Test RMSE', 'Test MAE', 'Test MAPE', 'CV R2 Mean']]
radar_df = radar_df.apply(normalize_series, axis=0)

# Plot radar chart
categories = list(radar_df.columns)
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)

for idx, row in radar_df.iterrows():
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=idx)
    ax.fill(angles, values, alpha=0.1)

plt.title("Model Performance Radar Chart\n(Normalized metrics, 1=best)", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

# =============================================
# Additional: Time Efficiency Comparison
# =============================================
import time

timing_results = []
for name, model in tuned_models.items():
    # Handle scaling
    if name in ['KNN', 'OLS']:
        X_tr = X_train_scaled
    else:
        X_tr = X_train

    # Training time
    start = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - start

    # Prediction time (on test set)
    start = time.time()
    model.predict(X_tr[:100])  # Predict on 100 samples
    pred_time = time.time() - start

    timing_results.append({
        'Model': name,
        'Training Time (s)': train_time,
        'Prediction Time/100 (s)': pred_time
    })

timing_df = pd.DataFrame(timing_results)
print("\n=== Computational Efficiency ===")
print(timing_df.sort_values('Training Time (s)'))

# =============================================
# Enhanced Learning Curves & Residual Analysis
# =============================================
from sklearn.model_selection import learning_curve
def enhanced_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1)

    plt.figure(figsize=(12,6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='b', label="Training score")
    plt.fill_between(train_sizes,
                    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                    alpha=0.1, color='b')

    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='r', label="CV score")
    plt.fill_between(train_sizes,
                    np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                    np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                    alpha=0.1, color='r')

    plt.title(f"{model_name} Learning Curve\n(Shaded areas show ±1 std. dev.)")
    plt.xlabel("Training examples")
    plt.ylabel("R2 Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def enhanced_residual_analysis(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, ax = plt.subplots(1, 3, figsize=(18,5))

    # Residuals vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[0])
    ax[0].axhline(y=0, color='r', linestyle='--')
    ax[0].set_title(f"{model_name} Residuals vs Predicted")
    ax[0].set_xlabel("Predicted Values")
    ax[0].set_ylabel("Residuals")

    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=ax[1])
    ax[1].set_title(f"{model_name} Residual Distribution")

    # Q-Q Plot
    sm.qqplot(residuals, line='s', ax=ax[2])
    ax[2].set_title(f"{model_name} Q-Q Plot")

    plt.tight_layout()
    plt.show()

    # Calculate residual statistics
    print(f"\n{model_name} Residual Analysis:")
    print(f"Mean residuals: {residuals.mean():.4f}")
    print(f"Residual std. dev.: {residuals.std():.4f}")
    print(f"Percentage of residuals within ±1 std. dev.: {(residuals.abs() <= residuals.std()).mean()*100:.2f}%")

# Run analyses for all five models
models_to_analyze = {
    'KNN': knn_cv.best_estimator_,
    'OLS': LinearRegression(),
    'Gradient Boosting': gb_cv.best_estimator_,
    'Random Forest': rf_cv.best_estimator_,
    'XGBoost': xgb_cv.best_estimator_
}

# Prepare scaled data for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add constant for OLS if using statsmodels
X_train_const = sm.add_constant(X_train_scaled)
X_test_const = sm.add_constant(X_test_scaled)

for name, model in models_to_analyze.items():
    print(f"\n{'='*40}\nAnalyzing: {name}\n{'='*40}")

    # Select proper input data
    if name in ['KNN', 'OLS']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # Learning curve
    enhanced_learning_curve(model, X_tr, y_train, name)

    # Residual analysis
    enhanced_residual_analysis(model, X_tr, y_train, X_te, y_test, name)

    # For OLS, add VIF analysis
    if name == 'OLS':
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif = pd.DataFrame()
        vif["Feature"] = ['Intercept'] + selected_features
        vif["VIF"] = [variance_inflation_factor(X_train_const, i) for i in range(X_train_const.shape[1])]
        print("\nVariance Inflation Factors (VIF):")
        print(vif.sort_values('VIF', ascending=False))

from sklearn.model_selection import cross_validate

# Define KFold (consistent with your setup)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store CV results
cv_results = {}

for model_name, model in tuned_models.items():
    # Handle scaling for KNN/OLS
    X_data = X_train_scaled if model_name in ['KNN', 'OLS'] else X_train

    # Get CV scores (using cross_validate for flexibility)
    scores = cross_validate(
        model,
        X_data,
        y_train,
        cv=cv,
        scoring='r2',
        return_train_score=False,
        return_estimator=False
    )

    # Store individual fold scores and summary stats
    cv_results[model_name] = {
        'Fold Scores': scores['test_score'],
        'Mean CV R2': np.mean(scores['test_score']),
        'Std CV R2': np.std(scores['test_score'])
    }

# Convert to DataFrame for clean display
cv_df = pd.DataFrame({
    model: [
        cv_results[model]['Fold Scores'],
        cv_results[model]['Mean CV R2'],
        cv_results[model]['Std CV R2']
    ]
    for model in cv_results.keys()
}, index=['Fold Scores (R2)', 'Mean CV R2', 'Std CV R2']).T

print("=== K-Fold Cross-Validation Results ===")
display(cv_df)

# Optional: Save to CSV
cv_df.to_csv('kfold_cv_results.csv')

"""##Correlation with Price##"""

numerical_data_corr = Melbourne1.select_dtypes(include=['int64', 'float64'])
corr_matrix = numerical_data_corr.corr()

print("\n--- Top Features Correlated with Price ---")
price_corr = corr_matrix[['price']].sort_values('price', ascending=False)
print(price_corr)

"""##Geographic and Relationship Visualizations##"""

# Geographic distribution by room type
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='longitude',
    y='latitude',
    hue='room_type',
    data=Melbourne1,
    palette='viridis',
    alpha=0.6
)
plt.title('Geographic Distribution by Room Type')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Getting the room details (Donut Chart)
roomdf = Melbourne1.groupby('room_type').size()/Melbourne1['room_type'].count()*100
labels = roomdf.index
values = roomdf.values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
fig.update_layout(title_text='Distribution of Room Types')
fig.show()

# Catplot room type and price
plt.figure(figsize=(10,6))
sns.catplot(x="room_type", y="price", data=Melbourne1);
plt.title('Price vs Room Type') # Add title to catplot
plt.ioff() # Turn off interactive mode, so plot doesn't show immediately if using it
plt.show() # Explicitly show the plot

# Word Cloud of Listing Names
text = " ".join(str(each) for each in Melbourne1.name.dropna()) # Ensure no NaNs are passed
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200, background_color="white").generate(text)
plt.figure(figsize=(15,10)) # Adjust figure size for better display
# Display the generated image:
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.title('Word Cloud of Listing Names') # Add title
plt.show()

# Property Type Distribution
prop_type_counts = Melbourne1['property_type'].value_counts(dropna=False)
print("\n--- Property Type Distribution ---")
print(prop_type_counts)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=prop_type_counts.head(20).values,
    y=prop_type_counts.head(20).index,
    palette='viridis'
)
plt.title('Top 20 Property Types')
plt.xlabel('Number of Listings')
plt.ylabel('Property Type')
plt.tight_layout()
plt.show()

# Price Distribution by Top 20 Property Types (with outlier removal for visualization)
# Get top 20 property types by frequency
top_20_properties = Melbourne1['property_type'].value_counts().nlargest(20).index

# Filter the data to include only top 20 property types and remove top 5% price outliers
filtered_data = Melbourne1[
    (Melbourne1['property_type'].isin(top_20_properties)) &
    (Melbourne1['price'] <= Melbourne1['price'].quantile(0.95))
]

# Create the plot
plt.figure(figsize=(12, 8))
sns.boxplot(
    y='property_type',
    x='price',
    data=filtered_data,
    palette='coolwarm',
    order=top_20_properties  # This ensures the property types are ordered by frequency
)
plt.title('Price Distribution by Top 20 Property Types (excluding top 5% price outliers)')
plt.ylabel('')
plt.xlabel('Price (AUD)')
plt.tight_layout()
plt.show()

# Room type breakdown by property type
print("\n--- Room Type Breakdown by Property Type ---")
display(pd.crosstab(
    Melbourne1['property_type'],
    Melbourne1['room_type'],
    margins=True
).sort_values('All', ascending=False))

# Geographic distribution by property type (Top 5)
if all(col in Melbourne1.columns for col in ['latitude', 'longitude']):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='longitude',
        y='latitude',
        hue='property_type',
        data=Melbourne1[Melbourne1['property_type'].isin(prop_type_counts.head(5).index)],
        palette='tab10',
        alpha=0.6
    )
    plt.title('Geographic Distribution of Top 5 Property Types')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
    plt.tight_layout()
    plt.show()

# Statistical summary by property type
print("\n--- Price Statistics by Property Type ---")
print(
    Melbourne1.groupby('property_type')['price']
    .describe(percentiles=[.25, .5, .75, .9])
    .sort_values('count', ascending=False)
)

# Room availability heatmap
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    x=Melbourne1['longitude'],
    y=Melbourne1['latitude'],
    c=Melbourne1['availability_365'],  # Color by room_availability
    cmap='viridis',
    edgecolor='black',
    linewidth=0.5,
    alpha=0.7
)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Room Availability (Days)')

# Add titles and labels
plt.title('Property Locations Colored by Room Availability')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Distribution of Availability
plt.figure(figsize=(10, 6))
sns.histplot(
    Melbourne1['availability_365'],
    bins=30,
    kde=True,
    color='skyblue'
)
plt.title('Distribution of Availability (Days/Year)')
plt.xlabel('Days Available per Year')
plt.ylabel('Count')
plt.axvline(
    Melbourne1['availability_365'].median(),
    color='red',
    linestyle='--',
    label=f'Median: {Melbourne1["availability_365"].median():.0f} days'
)
plt.legend()
plt.show()

plt.figure(figsize=(12, 10))

# Simple geographic plot
sns.scatterplot(
    x='longitude',
    y='latitude',
    data=Melbourne1,
    color='royalblue',  # Single color for all points
    alpha=0.5,
    s=10  # Point size
)

# Add context
plt.title('Geographic Distribution of Airbnb Listings in Sydney', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

# Set proper aspect ratio for geographic data
ax = plt.gca()
ax.set_aspect(1.2)  # Adjust this value based on your coordinate range

# Add grid
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

import folium
from folium.plugins import MarkerCluster

# Calculate center point of all listings
center_lat = Melbourne1['latitude'].mean()
center_lon = Melbourne1['longitude'].mean()

# Create base map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add marker clusters for better visualization of dense areas
marker_cluster = MarkerCluster().add_to(m)

# Add each listing to the map
for idx, row in Melbourne1.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='royalblue',
        fill=True,
        fill_opacity=0.5,
        popup=f"Price: ${row['price']}",
        tooltip=row['neighbourhood_cleansed']
    ).add_to(marker_cluster)

# Add tile layer for better map styling
folium.TileLayer('cartodbpositron').add_to(m)  # Light theme
folium.TileLayer('openstreetmap').add_to(m)     # Standard OSM
folium.LayerControl().add_to(m)  # Allows toggling between map styles

# Add title
title_html = '''
    <h3 style="font-size:16px; margin-top:10px">
    <b>Geographic Distribution of Airbnb Listings in Sydney</b>
    </h3>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Display the map
display(m)

# Save to HTML file
m.save('sydney_airbnb_map.html')

"""##Geographic Feature Engineering ##"""

import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.neighbors import BallTree

# Create base map centered on Sydney
m = folium.Map(location=[-33.8688, 151.2093], zoom_start=11)

# Create heatmap layer
# Ensure 'latitude' and 'longitude' are not NaN for heatmap
heat_data = Melbourne1[['latitude', 'longitude']].dropna().values.tolist()

HeatMap(
    heat_data,
    radius=8,
    blur=10,
    gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'orange', 1.0: 'red'}
).add_to(m)

print("\n--- Interactive Heatmap of Listing Density in Sydney ---")
display(m)

# Distance to Sydney CBD (using Sydney Town Hall as reference point)
CBD_COORD = (-33.8738, 151.2069)

def calculate_distance(lat, lon, ref_point=CBD_COORD):
    """Calculate distance in kilometers from reference point."""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    return geodesic((lat, lon), ref_point).km

# Add distance to CBD column
Melbourne1['distance_to_cbd'] = Melbourne1.apply(
    lambda x: calculate_distance(x['latitude'], x['longitude']), axis=1
)
print("\n--- Distance to CBD Head ---")
print(Melbourne1['distance_to_cbd'].head())

# Distance to major Sydney landmarks
SYDNEY_LANDMARKS = {
    'sydney_airport': (-33.9399, 151.1753),
    'central_station': (-33.8832, 151.2066),
    'university_of_sydney': (-33.8884, 151.1873),
    'opera_house': (-33.8568, 151.2153),
    'bondi_beach': (-33.8915, 151.2767),
    'manly_beach': (-33.8000, 151.2887),
    'darling_harbour': (-33.8747, 151.1989),
    'taronga_zoo': (-33.8466, 151.2419)
}

# Calculate distances to each landmark
for name, coords in SYDNEY_LANDMARKS.items():
    Melbourne1[f'distance_to_{name}'] = Melbourne1.apply(
        lambda x: calculate_distance(x['latitude'], x['longitude'], coords), axis=1
    )
print("\n--- Distance to Landmarks Head ---")
print(Melbourne1[['distance_to_sydney_airport', 'distance_to_opera_house', 'distance_to_bondi_beach']].head())

# Listing density features using BallTree
coords_for_tree = Melbourne1[['latitude', 'longitude']].dropna()

if not coords_for_tree.empty:
    # Convert to radians for haversine metric
    coords_rad = np.deg2rad(coords_for_tree)
    tree = BallTree(coords_rad, metric='haversine')

    # Calculate listings within various radii
    for radius in [0.5, 1, 2, 5]:  # in kilometers
        rad_val = radius / 6371  # Convert km to radians
        listings = pd.Series(np.nan, index=Melbourne1.index)
        listings.loc[coords_for_tree.index] = tree.query_radius(
            coords_rad, r=rad_val, count_only=True
        )
        Melbourne1[f'listings_within_{radius}km'] = listings
        print(f"\nAdded listings within {radius}km radius")

    print("\n--- Listings Density Features Head ---")
    print(Melbourne1[['listings_within_0.5km', 'listings_within_1km', 'listings_within_2km']].head())
else:
    print("Warning: No valid coordinates found for density calculations")

# Additional geographic features
def add_geographic_features(df):
    """Add derived geographic features"""
    # Coastal proximity flag (within 2km of major beaches)
    coastal_landmarks = ['bondi_beach', 'manly_beach']
    df['near_coast'] = df.apply(
        lambda x: any(x[f'distance_to_{lm}'] <= 2 for lm in coastal_landmarks
                      if not pd.isna(x[f'distance_to_{lm}'])), axis=1)

    # Transportation hub proximity
    df['near_transport_hub'] = df.apply(
        lambda x: x['distance_to_sydney_airport'] <= 10 or
                 x['distance_to_central_station'] <= 2, axis=1)

    # Tourist area flag
    tourist_landmarks = ['opera_house', 'darling_harbour', 'taronga_zoo']
    df['in_tourist_area'] = df.apply(
        lambda x: any(x[f'distance_to_{lm}'] <= 3 for lm in tourist_landmarks
                     if not pd.isna(x[f'distance_to_{lm}'])), axis=1)

    return df

Melbourne1 = add_geographic_features(Melbourne1)

print("\n--- Final DataFrame with Geographic Features ---")
print(Melbourne1.info())
print("\n--- Sample of Geographic Features ---")
print(Melbourne1[['distance_to_cbd', 'near_coast', 'near_transport_hub', 'in_tourist_area']].sample(5))

"""##Neighbourhood analysis##"""

# Create working dataset with only known neighborhoods
known_neighborhoods = Melbourne1[Melbourne1['host_neighbourhood'] != 'Unknown'].copy()

# Top neighborhoods by listings count
print("\n--- Top 10 Neighborhoods by Listings Count ---")
top_neighborhoods = known_neighborhoods['host_neighbourhood'].value_counts().nlargest(10)
print(top_neighborhoods)

# Neighborhood performance metrics
print("\n--- Neighborhood Performance Metrics ---")
neighborhood_stats = known_neighborhoods.groupby('host_neighbourhood').agg({
    'price': ['median', 'count'],
    'review_scores_rating': 'median',
    'host_is_superhost': 'mean',
    'number_of_reviews': 'median'
}).sort_values(('price', 'median'), ascending=False)
print(neighborhood_stats.head(10).to_string())

# ANOVA test for price differences among top 5 neighborhoods
top_5_neigh = known_neighborhoods['host_neighbourhood'].value_counts().nlargest(5).index
groups_anova = [known_neighborhoods[known_neighborhoods['host_neighbourhood']==n]['price']
                for n in top_5_neigh]
f_stat_anova, p_value_anova = f_oneway(*groups_anova)
print(f"\n--- ANOVA p-value for price differences (Top 5 Neighborhoods): {p_value_anova:.4f} ---")

"""##Demand Forecasting##"""

# Seasonal demand visualization
Melbourne1['last_review'] = pd.to_datetime(Melbourne1['last_review'])
Melbourne1['month'] = Melbourne1['last_review'].dt.month
monthly_demand = Melbourne1.groupby('month')['number_of_reviews'].sum()

# Booking trends by month and season
Melbourne1['review_month'] = Melbourne1['last_review'].dt.month
Melbourne1['review_year'] = Melbourne1['last_review'].dt.year

seasonal_trends = Melbourne1.groupby('review_month')['number_of_reviews'].sum().reset_index()

season_map = {
    1: 'Summer', 2: 'Summer', 3: 'Autumn',
    4: 'Autumn', 5: 'Autumn', 6: 'Winter',
    7: 'Winter', 8: 'Winter', 9: 'Spring',
    10: 'Spring', 11: 'Spring', 12: 'Summer'
}
seasonal_trends['season'] = seasonal_trends['review_month'].map(season_map)

plt.figure(figsize=(10, 5))
sns.lineplot(x='review_month', y='number_of_reviews', data=seasonal_trends, marker='o')
plt.title("Airbnb Booking Trends by Month")
plt.xlabel("Month")
plt.ylabel("Total Reviews (Proxy for Bookings)")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()

print("\n--- Seasonal Performance ---")
seasonal_performance = seasonal_trends.groupby('season')['number_of_reviews'].sum().reset_index()
print(seasonal_performance)

"""##Host Performance##"""

# Superhost classifier feature importance
X_superhost = Melbourne1[['host_response_rate_num', 'host_acceptance_rate_num', 'review_scores_rating']].dropna()
y_superhost = Melbourne1.loc[X_superhost.index]['host_is_superhost']

clf = RandomForestClassifier(random_state=42)
clf.fit(X_superhost, y_superhost)

plt.figure(figsize=(8, 5))
pd.Series(clf.feature_importances_, index=X_superhost.columns).plot(kind='barh')
plt.title('Superhost Classifier Feature Importance')
plt.show()

"""##Geographic Analysis##"""

# Base map with CBD and top 50 properties by price
Syd_map = folium.Map(location=CBD_COORD, zoom_start=12)

folium.Marker(
    CBD_COORD,
    popup='Sydney CBD',
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(Syd_map)

for _, row in Melbourne1.nlargest(50, 'price').iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='blue',
        fill=True,
        popup=f"${row['price']}/night, {row['distance_to_cbd']:.1f}km to CBD"
    ).add_to(Syd_map)
print("\n--- Interactive Map with CBD and Top 50 Properties (displayed below) ---")
display(Syd_map)

"""##Operational Efficiency##"""

# Occupancy-Price tradeoff
plt.figure(figsize=(10, 6))
sns.scatterplot(x='availability_365', y='price',
                hue='room_type', data=Melbourne1, alpha=0.6)
plt.title('Price vs Availability')
plt.show()

# Average Listing Price by Host Join Year
Melbourne1['host_since_year'] = pd.to_datetime(Melbourne1['host_since']).dt.year
plt.figure(figsize=(12,6))
sns.lineplot(
    x='host_since_year',
    y='price',
    data=Melbourne1,
    estimator='mean',
    errorbar=None,
    marker='o'
)
plt.title('Average Listing Price by Host Join Year')
plt.xlabel('Year Host Joined')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Average Listing Price by Host Join Month
Melbourne1['host_since_month'] = pd.to_datetime(Melbourne1['host_since']).dt.month
plt.figure(figsize=(12,6))
sns.lineplot(
    x='host_since_month',
    y='price',
    data=Melbourne1,
    estimator='mean',
    errorbar=None,
    marker='o'
)
plt.title('Average Listing Price by Host Join Month')
plt.xlabel('Month Host Joined')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Distribution of guest capacity
plt.figure(figsize=(10, 6))
sns.histplot(
    data=Melbourne1,
    x='accommodates',
    bins=range(1, int(Melbourne1['accommodates'].max())+2), # Ensure bins cover all integers
    kde=False,
    color='royalblue'
)
plt.title('Distribution of Guest Capacity')
plt.xlabel('Number of Guests Accommodated')
plt.ylabel('Number of Listings')
plt.xticks(range(1, int(Melbourne1['accommodates'].max())+1))
plt.grid(axis='y', alpha=0.3)
plt.show()

Melbourne1.info()

"""##Review Analysis##"""

# Distribution of number of reviews
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(Melbourne1['number_of_reviews'], bins=50, kde=True)
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')

plt.subplot(1, 2, 2)
sns.boxplot(x=Melbourne1['number_of_reviews'])
plt.title('Boxplot of Review Counts')
plt.tight_layout()
plt.show()

print("\n--- Statistical Summary of Number of Reviews ---")
print(Melbourne1['number_of_reviews'].describe())

# Price vs Number of Reviews
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.regplot(x='number_of_reviews', y='price',
            data=Melbourne1,
            scatter_kws={'alpha':0.3},
            line_kws={'color':'red'})
plt.title('Price vs Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Price ($)')

plt.subplot(1, 2, 2)
bins_reviews = [0, 5, 10, 20, 50, 100, 200, 500, Melbourne1['number_of_reviews'].max()]
Melbourne1['review_bins'] = pd.cut(Melbourne1['number_of_reviews'], bins=bins_reviews, right=False)
sns.boxplot(x='review_bins', y='price', data=Melbourne1)
plt.title('Price Distribution by Review Count Ranges')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Drill down into strong relationships (accommodates vs price)
sns.jointplot(x='accommodates', y='price', data=Melbourne1, kind='reg')
plt.show()

# Check for nonlinear relationship (number_of_reviews vs price)
sns.lmplot(x='number_of_reviews', y='price',
           data=Melbourne1,
           lowess=True,
           scatter_kws={'alpha':0.3})
plt.show()

"""##PCA Principal Component Analysis##"""

#install the missing package
!pip install kneed

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import plotly.express as px
from kneed import KneeLocator

#######################################
# 1. PCA with Biplot Visualization
#######################################
# Select and prepare features
features = ['bedrooms', 'longitude', 'latitude', 'bathrooms',
    'accommodates', 'review_scores_location', 'beds',
    'availability_365', 'host_total_listings_count',
    'reviews_per_month'
]

X = Melbourne1[features].fillna(Melbourne1[features].median())

def run_pca_analysis(X, n_components=2, pipeline=None):
    """Perform PCA and create biplot with loadings, with option to use existing pipeline"""
    if pipeline is None:
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
    else:
        # Use existing pipeline components
        scaler = pipeline.named_steps['scaler']
        pca = pipeline.named_steps['reducer']
        X_scaled = scaler.transform(X)
        principal_components = pca.transform(X_scaled)

    # Create results DataFrame
    pca_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC{i+1}' for i in range(n_components)])

    # Create biplot
    plt.figure(figsize=(15, 10))

    # Scatter plot of observations
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.6)

    # Plot feature loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    features = X.columns

    for i, feature in enumerate(features):
        plt.arrow(0, 0, loadings[i,0]*0.8, loadings[i,1]*0.8,
                 color='r', alpha=0.5, head_width=0.05)
        plt.text(loadings[i,0]*0.85, loadings[i,1]*0.85,
                feature, color='r', ha='center', va='center')

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("PCA Biplot with Feature Loadings")
    plt.grid()

    # Scree plot
    plt.subplot(2, 2, 2)
    bars = plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
             np.cumsum(pca.explained_variance_ratio_),
             color='red', marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')

    # Print key variables
    print("\nKey Variables by PC:")
    for i in range(n_components):
        print(f"\nPC{i+1} Top Variables:")
        sorted_vars = sorted(zip(features, pca.components_[i]),
                         key=lambda x: abs(x[1]), reverse=True)[:5]
        for var, loading in sorted_vars:
            print(f"{var}: {loading:.3f}")

    # Component loadings table
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=features
    )
    print("\nComponent Loadings:")
    display(loadings_df)

    # Variance analysis
    print("\nVariance Analysis:")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", sum(pca.explained_variance_ratio_))

    return pca, pca_df, X_scaled

#######################################
# 2. K-means Clustering on PCA Scores
#######################################

def optimize_kmeans(data, max_k=10):
    """Determine optimal k using elbow and silhouette methods"""
    inertias = []
    silhouettes = []

    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

        if k > 1:
            silhouettes.append(silhouette_score(data, kmeans.labels_))

    # Elbow plot
    kneedle = KneeLocator(range(2,max_k+1), inertias, curve='convex', direction='decreasing')
    plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)
    plt.plot(range(2,max_k+1), inertias, marker='o')
    if kneedle.elbow:
        plt.axvline(kneedle.elbow, color='r', linestyle='--')
    plt.title(f"Elbow Method (Suggested k={kneedle.elbow or 'Not clear'})")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")

    # Silhouette plot
    plt.subplot(1,2,2)
    plt.plot(range(2,max_k+1), silhouettes, marker='o')
    if len(silhouettes) > 0:
        optimal_k = np.argmax(silhouettes) + 2
        plt.axvline(optimal_k, color='r', linestyle='--')
    plt.title(f"Silhouette Method (Suggested k={optimal_k if 'optimal_k' in locals() else 'Not clear'})")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()

    return kneedle.elbow if kneedle.elbow else optimal_k

##############################
# 3. Hierarchical Clustering
##############################

def hierarchical_clustering(data, method='ward'):
    """Generate dendrogram for hierarchical clustering"""
    plt.figure(figsize=(12,6))
    linked = linkage(data, method=method)
    dendrogram(linked, orientation='top',
              distance_sort='descending',
              show_leaf_counts=True)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

#######################################
# 4. Cluster Interpretation & Insights
#######################################

def interpret_clusters(pca_obj, pca_scores, clusters, original_df, n_features=5):
    """Generate cluster profiles and actionable insights"""
    # Create cluster profiles
    cluster_df = original_df.copy()
    cluster_df['Cluster'] = clusters

    # 1. Profile Table (Means/Medians)
    profile_table = cluster_df.groupby('Cluster').agg(['mean', 'median'])
    print("\nCluster Profiles:")
    display(profile_table)

    # 2. 2D Visualization with Centroids
    centroids = pd.DataFrame(pca_obj.transform(
        cluster_df.groupby('Cluster').mean()),
        columns=[f'PC{i+1}' for i in range(pca_scores.shape[1])])

    plt.figure(figsize=(10,8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster',
                   data=pca_scores.assign(Cluster=clusters),
                   palette='viridis', alpha=0.6)
    sns.scatterplot(x='PC1', y='PC2', data=centroids,
                   color='black', marker='X', s=200, label='Centroids')
    plt.title("Cluster Visualization with Centroids")
    plt.legend()
    plt.show()

    # 3. Key Drivers per Cluster
    print("\nKey Drivers per Cluster:")
    for cluster in np.unique(clusters):
        print(f"\nCluster {cluster}:")
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster].drop('Cluster', axis=1)
        overall_means = original_df.mean()

        # Find most distinctive features
        ratios = (cluster_data.mean() / overall_means).sort_values(ascending=False)
        print("Most elevated features:")
        print(ratios.head(n_features))

        print("\nMost depressed features:")
        print(ratios.tail(n_features))

    # 4. Cross-tabulation with InGroup (if exists)
    if 'InGroup' in original_df.columns:
        print("\nCluster vs InGroup Crosstab:")
        crosstab = pd.crosstab(cluster_df['Cluster'], original_df['InGroup'],
                              normalize='index')
        display(crosstab.style.background_gradient(cmap='Blues'))

        # Interpretation guidance
        print("\nInsights:")
        for cluster in crosstab.index:
            dominant_group = crosstab.loc[cluster].idxmax()
            print(f"Cluster {cluster} is {crosstab.loc[cluster, dominant_group]:.1%} aligned with {dominant_group}")

#######################################
# Main Execution Pipeline
#######################################

# 1. Run PCA
pca, pca_scores, X_scaled = run_pca_analysis(X)

# 2. Determine optimal k
optimal_k = optimize_kmeans(pca_scores)

# 3. Perform K-means on PCA scores
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_scores)

# 4. Hierarchical clustering
hierarchical_clustering(pca_scores)

# 5. Generate insights
interpret_clusters(pca, pca_scores, clusters, X)

# 6. Interactive 3D visualization
if pca_scores.shape[1] >= 3:
    fig = px.scatter_3d(pd.DataFrame(pca_scores, columns=['PC1','PC2','PC3']).assign(Cluster=clusters),
                       x='PC1', y='PC2', z='PC3', color='Cluster',
                       title="3D Cluster Visualization")
    fig.show()

"""##Model Interpretability XAi##"""

import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# 1. Train XGBoost (replace with your actual data)
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)  # Use your pre-split data

# 2. Global Explanations (Feature Importance)
def global_explanations(model, X, y):
    """SHAP summary plot and permutation importance"""
    plt.figure(figsize=(15, 6))

    # SHAP Summary Plot
    plt.subplot(1, 2, 1)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Global Feature Impact", fontsize=12)

    # Permutation Importance
    plt.subplot(1, 2, 2)
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X.columns[sorted_idx])
    plt.title("Permutation Importance", fontsize=12)
    plt.tight_layout()
    plt.show()

    return shap_values

# 3. Local Explanations (Single Prediction)
def local_explanations(model, X, instance_idx):
    """LIME and SHAP force plot for one prediction"""
    # LIME
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values, feature_names=X.columns, mode='regression'
    )
    lime_exp = lime_explainer.explain_instance(
        X.iloc[instance_idx], model.predict, num_features=5
    )

    # SHAP Force Plot
    shap_explainer = shap.Explainer(model)
    shap_values = shap_explainer(X.iloc[[instance_idx]])

    # Plot both
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    lime_exp.as_pyplot_figure()
    plt.title(f"LIME: Instance {instance_idx}")

    plt.subplot(1, 2, 2)
    shap.plots.force(
        shap_explainer.expected_value,
        shap_values.values[0],
        X.iloc[instance_idx],
        matplotlib=True
    )
    plt.title(f"SHAP: Instance {instance_idx}")
    plt.tight_layout()
    plt.show()

# Run XAI
print("=== Global Explanations ===")
shap_values = global_explanations(model, X_test, y_test)

print("\n=== Local Explanation (First Test Instance) ===")
local_explanations(model, X_test, 0)

# Advanced: SHAP Dependence Plot (for top feature)
print("\n=== SHAP Dependence Plot ===")
shap.plots.scatter(shap_values[:, 0], color=shap_values)
Appendix 2: Sydney 
# -*- coding: utf-8 -*-
"""SydneyDataset.ipynb
Automatically generated by Colab.
Original file is located at
    https://colab.research.google.com/drive/1smWQMSv0WvSA1CDlvspvjffWfBSCyLqw
"""##Load Dataset##"""
Sydney = pd.read_csv("/content/SydneyListings.csv")
"""##Initial Exploratory Data Analysis##"""
print(Sydney.head())
print(Sydney.info())
print(Sydney['last_scraped'])
"""##Descriptive Statistics for numerical columns##"""

descriptive_stats=Sydney.describe()
print(descriptive_stats)
# Get numerical columns (int and float)
num_cols = Sydney.select_dtypes(include=['int64', 'float64']).columns
n_cols = len(num_cols)
# Calculate grid size dynamically
n_rows = (n_cols // 4) + (1 if n_cols % 4 != 0 else 0)  # 4 columns per row
plt.figure(figsize=(20, 5 * n_rows))  # Adjust height based on rows
for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, 4, i)
    sns.histplot(data=Sydney, x=col, bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
# Visualization for categorical variables
# Identify categorical columns (excluding any list-type columns)
cat_cols = [col for col in Sydney.select_dtypes(include=['object', 'category']).columns
            if not any(isinstance(x, list) for x in Sydney[col].head())]
# Now create the visualization
plt.figure(figsize=(20, 10))
for i, col in enumerate(cat_cols[:8]):  # First 8 valid categorical columns
    plt.subplot(4, 2, i+1)
    # Convert to string if needed (safe handling)
    if Sydney[col].apply(type).eq(list).any():
        plot_data = Sydney[col].astype(str).value_counts().head(10)
    else:
        plot_data = Sydney[col].value_counts().head(10)
    plot_data.plot(kind='bar')
    plt.title(f'Distribution of {col}', pad=10)
    plt.xticks(rotation=45)
    plt.xlabel('')
plt.tight_layout()
plt.show()
"""##Initial Data Exploration##"""
#Check for missing values
missing_values=Sydney.isnull().sum()
print(missing_values)
#Check for duplicate rows
duplicate_rows= Sydney.duplicated().sum()
print(duplicate_rows)
#Summary of data types
data_types= Sydney.dtypes
print(data_types)
"""##Data Cleaning: Type Conversion and Initial Formatting##"""
# Convert price to numeric (remove $ and commas)
Sydney['price'] = Sydney['price'].str.replace('[$,]', '', regex=True).astype(float)
print("\n--- Price Column Head (After Cleaning) ---")
print(Sydney['price'].head())
# Convert boolean columns
Sydney['host_is_superhost'] = Sydney['host_is_superhost'].map({'t': 1, 'f': 0})
Sydney['instant_bookable'] = Sydney['instant_bookable'].map({'t': 1, 'f': 0})
print("\n--- Superhost Status Descriptive Statistics ---")
print(Sydney['host_is_superhost'].describe())
print("\n--- Instant Bookable Status Descriptive Statistics ---")
print(Sydney['instant_bookable'].describe())
# Convert date columns
Sydney['host_since'] = pd.to_datetime(Sydney['host_since'])
Sydney['last_review'] = pd.to_datetime(Sydney['last_review'])
Sydney['first_review'] = pd.to_datetime(Sydney['first_review'])
# Extract year from host_since for analysis
Sydney['year'] = Sydney['host_since'].dt.year
"""Keep relevant variables"""
# List of relevant columns for price modeling
relevant_columns = [
    'id', 'name', 'host_id', 'host_name', 'host_since', 'host_is_superhost', 'neighbourhood', 'neighbourhood_cleansed', 'host_neighbourhood', 'host_total_listings_count', 'latitude', 'longitude', 'property_type',
    'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 'maximum_nights', 'availability_365',
    'number_of_reviews', 'number_of_reviews_ltm', 'first_review', 'last_review', 'host_response_rate', 'host_acceptance_rate', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value', 'instant_bookable', 'calculated_host_listings_count', 'reviews_per_month', 'year'
]
# Filter the dataframe
Sydney1 = Sydney[relevant_columns].copy()
print("\n--- DataFrame Info After Selecting Relevant Columns ---")
Sydney1.info()

batch_size = 12
num_cols = list(Sydney1.select_dtypes(include=['int64', 'float64']).columns)
for batch in range(0, len(num_cols), batch_size):
    plt.figure(figsize=(20, 15))
    current_cols = num_cols[batch:batch+batch_size]
    for i, column in enumerate(current_cols, 1):
        plt.subplot(4, 3, i)
        sns.histplot(Sydney1[column], kde=True, color='brown')
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()
# Drop Price NA value
Sydney1.dropna(subset=['price'], inplace=True)
print("\n--- Price Column Head (After Dropping NA) ---")
print(Sydney1['price'].head())
print("\n--- Price Descriptive Statistics (After Cleaning) ---")
print(Sydney1['price'].describe())
# Drop NaN values in host_since
print("\n--- Before Dropping host_since NA Values ---")
print(f"Total rows: {len(Sydney1)}")
print(f"Missing host_since: {Sydney1['host_since'].isna().sum()}")
Sydney1.dropna(subset=['host_since'], inplace=True)
print("\n--- After Dropping host_since NA Values ---")
print(f"Remaining rows: {len(Sydney1)}")
print(f"Earliest host since: {Sydney1['host_since'].min()}")
print(f"Latest host since: {Sydney1['host_since'].max()}")
"""##Data Cleaning##"""
# Calculate missing values
missing = Sydney1.isnull().sum()
missing_percent = (missing / len(Sydney1)) * 100
print("\n--- Missing Value Summary (Before Imputation) ---")
print(pd.DataFrame({'Missing Count': missing, 'Missing Percentage': missing_percent}).sort_values('Missing Count', ascending=False))
# Create a missing data matrix
plt.figure(figsize=(12, 8))
sns.heatmap(Sydney1.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Matrix (Yellow = Missing)')
plt.xticks(rotation=90)
plt.show()
# Checking for outliers
num_cols = list(Sydney1.select_dtypes(include=['int64', 'float64']).columns)
## Function to identify outliers using IQR method
def identify_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
# Identifying outliers for each numerical column
outliers_summary = {}
for col in num_cols:
    outliers = identify_outliers_iqr(Sydney1, col)
    outliers_summary[col] = len(outliers)
outliers_summary
# Create boxplots for each numerical column
plt.figure(figsize=(15, 8))
sns.set(style="whitegrid")
# Calculate number of rows needed for subplots
n_cols = 7  # Number of columns in subplot grid
n_rows = (len(num_cols) + n_cols - 1) // n_cols  # Round up division
# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
fig.suptitle('Boxplots of Numerical Columns with Outliers', y=1.02)
# Flatten axes array for easy iteration
axes = axes.flatten()
for i, col in enumerate(num_cols):
    if i < len(axes):  # Ensure we don't exceed number of subplots
        sns.boxplot(x=Sydney1[col], ax=axes[i], color='skyblue')
        axes[i].set_title(f'{col}\n(Outliers: {outliers_summary[col]})')
        axes[i].set_xlabel('')
# Hide any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
"""##Detailed Data Cleaning and Imputation##"""
cols_to_check = [
    'host_since', 'host_is_superhost', 'bathrooms',
    'bedrooms', 'beds', 'instant_bookable'
]
# Add review-related columns
review_cols = [col for col in Sydney1.columns if col.startswith('review_')]
cols_to_check.extend(review_cols)

# Create a summary DataFrame
missing_analysis = pd.DataFrame({
    'total_rows': len(Sydney1),
    'non_missing': Sydney1[cols_to_check].count(),
    'missing': Sydney1[cols_to_check].isna().sum(),
    'zero_values': (Sydney1[cols_to_check] == 0).sum()
})
missing_analysis['missing_pct'] = (missing_analysis['missing'] / missing_analysis['total_rows']).round(3)
missing_analysis['zero_pct'] = (missing_analysis['zero_values'] / missing_analysis['total_rows']).round(3)
print(missing_analysis[['total_rows', 'non_missing', 'missing', 'missing_pct', 'zero_values', 'zero_pct']])
# ==============
# Detailed DATA CLEANING
# ===============
# Outlier Treatment
print("\n=== OUTLIER HANDLING ===")
# Price Outliers
price_upper_limit = Sydney1['price'].quantile(0.99)
Sydney1 = Sydney1[Sydney1['price'] <= price_upper_limit].copy()
# Visual Validation
plt.figure(figsize=(12, 5))
sns.boxplot(x=Sydney1['price'])
plt.title('Price Distribution After Outlier Removal')
plt.show()
print("\nPrice Statistics After Cleaning:")
print(Sydney1['price'].describe(percentiles=[.25, .5, .75, .9, .95, .99]))
# Other Numeric Features
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
for col in ['beds', 'bedrooms', 'bathrooms']:
    Sydney1 = remove_outliers(Sydney1, col)
# Amenities Processing
print("\n=== AMENITIES PROCESSING ===")
Sydney1['amenities'] = Sydney1['amenities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
# Top Amenities Visualization
all_amenities = [amenity for sublist in Sydney1['amenities'] for amenity in sublist]
amenity_counts = Counter(all_amenities)
amenity_df = pd.DataFrame.from_dict(amenity_counts, orient='index', columns=['count']).sort_values('count', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y=amenity_df.index[:20], data=amenity_df[:20], palette='viridis')
plt.title('Top 20 Most Common Amenities')
plt.tight_layout()
plt.show()
# Missing Value Imputation
print("\n=== MISSING VALUE IMPUTATION ===")
# Categorical Columns
for col in ['host_neighbourhood', 'neighbourhood', 'host_name']:
    Sydney1[col] = Sydney1[col].fillna('unknown')
# Boolean Columns
Sydney1['host_is_superhost'] = Sydney1['host_is_superhost'].fillna(
    Sydney1['host_is_superhost'].mode()[0]
)
# Numeric Columns
for col in ['bathrooms', 'bedrooms', 'beds']:
    Sydney1[col] = Sydney1[col].fillna(Sydney1[col].mode()[0])
# Review Scores
review_columns = ['review_scores_rating', 'review_scores_accuracy',
                 'review_scores_cleanliness', 'review_scores_checkin',
                 'review_scores_communication', 'review_scores_location',
                 'review_scores_value']
for col in review_columns:
    Sydney1[col] = Sydney1[col].fillna(Sydney1[col].median())
# Date Columns
for col in ['host_since', 'last_review', 'first_review']:
    Sydney1[col] = Sydney1[col].fillna(Sydney1[col].median())
# Other Numeric Columns
for col in ['host_total_listings_count', 'reviews_per_month',
           'host_response_rate', 'host_acceptance_rate']:
    if Sydney1[col].dtype == 'object':
        Sydney1[col] = Sydney1[col].astype(str).str.replace('%', '').astype(float) / 100
    Sydney1[col] = Sydney1[col].fillna(Sydney1[col].mode()[0])
# Final Data Validation
print("\n=== FINAL DATA VALIDATION ===")
# Missing Values Check
print("\nRemaining Missing Values:")
print(Sydney1.isnull().sum().sort_values(ascending=False))
# Host Analysis
print("\n--- Host Information ---")
print(f"Date Range: {Sydney1['host_since'].min()} to {Sydney1['host_since'].max()}")
print(f"Superhost Distribution:\n{Sydney1['host_is_superhost'].value_counts()}")
# Neighborhood Analysis
plt.figure(figsize=(15, 7))
top_neighborhoods = Sydney1['host_neighbourhood'].value_counts().index[:15]
sns.countplot(x="host_neighbourhood", data=Sydney1, order=top_neighborhoods)
plt.title('Top 15 Neighborhoods by Listing Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(f"Final Dataset Shape: {Sydney1.shape}")
print("\n--- Final Descriptive Statistics ---")
print(Sydney1.describe())
# Check for missing values in neighbourhood_cleansed column
missing_count = Sydney1['neighbourhood_cleansed'].isnull().sum()
total_count = len(Sydney1['neighbourhood_cleansed'])
print(f"Missing values in 'neighbourhood_cleansed': {missing_count}")
print(f"Total records: {total_count}")
print(f"Percentage missing: {missing_count/total_count*100:.2f}%")
# Visual representation
plt.figure(figsize=(8, 5))
sns.heatmap(Sydney1[['neighbourhood_cleansed']].isnull(),
            yticklabels=False,
            cbar=False,
            cmap='viridis')
plt.title('Missing Values in neighbourhood_cleansed')
plt.show()
# Check for different types of missing/unknown values
na_count = Sydney1['neighbourhood_cleansed'].isna().sum()
null_count = Sydney1['neighbourhood_cleansed'].isnull().sum()
none_count = (Sydney1['neighbourhood_cleansed'] == None).sum()
unknown_count = (Sydney1['neighbourhood_cleansed'].str.lower() == 'unknown').sum()
blank_count = (Sydney1['neighbourhood_cleansed'] == '').sum()
print("Missing/Unknown Value Analysis:")
print(f"- NA values: {na_count}")
print(f"- Null values: {null_count}")
print(f"- None values: {none_count}")
print(f"- 'Unknown' text values: {unknown_count}")
print(f"- Blank strings: {blank_count}")
# Combined missing/unknown percentage
total_missing = na_count + unknown_count + blank_count
total_records = len(Sydney1)
print(f"\nTotal problematic values: {total_missing} ({total_missing/total_records*100:.2f}%)")
# Visual inspection
plt.figure(figsize=(10, 6))
missing_data = Sydney1['neighbourhood_cleansed'].isna() | \
               (Sydney1['neighbourhood_cleansed'].str.lower() == 'unknown') | \
               (Sydney1['neighbourhood_cleansed'] == '')
sns.heatmap(missing_data.to_frame(),
            yticklabels=False,
            cbar=False,
            cmap='Reds')
plt.title('Missing/Unknown Values in Neighbourhood_Cleansed')
plt.show()
# Show sample of problematic rows
problem_rows = Sydney1[missing_data]
if not problem_rows.empty:
    print("\nSample of problematic rows:")
    print(problem_rows[['neighbourhood_cleansed']].head())
else:
    print("\nNo missing or unknown values found!")
# Calculate summary statistics
min_nights_stats = Sydney1['minimum_nights'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print("Minimum Nights Statistics:")
print(min_nights_stats)
max_nights_stats = Sydney1['maximum_nights'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print("Maximum Nights Statistics:")
print(max_nights_stats)
plt.figure(figsize=(12, 6))
# Box plot
plt.subplot(1, 2, 1)
sns.boxplot(x=Sydney1['minimum_nights'])
plt.title('Box Plot of Minimum Nights')
# Distribution plot (log scale often helps)
plt.subplot(1, 2, 2)
sns.histplot(Sydney1['minimum_nights'], bins=50, log_scale=True)
plt.title('Distribution of Minimum Nights (Log Scale)')
plt.tight_layout()
plt.show()
# Calculate correlations
correlation_matrix = Sydney1[['minimum_nights', 'maximum_nights', 'price']].corr()
print("Correlation Matrix:")
print(correlation_matrix)
# Visualize with heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Price vs. Min/Max Nights Correlation")
plt.show()

# Categorize minimum nights
bins = [0, 1, 3, 7, 30, 365, np.inf]
labels = ['1 night', '2-3 nights', '4-7 nights', '1-4 weeks', '1-12 months', '1yr+']
Sydney1['min_stay_group'] = pd.cut(Sydney1['minimum_nights'], bins=bins, labels=labels)
# Plot price distribution
plt.figure(figsize=(12, 6))
sns.boxplot(x='min_stay_group', y='price', data=Sydney1, showfliers=False)
plt.title("Price Distribution by Minimum Stay Requirement")
plt.xticks(rotation=45)
plt.show()
# Categorize maximum nights
Sydney1['max_stay_group'] = pd.cut(Sydney1['maximum_nights'],
                                  bins=[0, 7, 30, 90, 365, np.inf],
                                  labels=['<1 week', '1-4 weeks', '1-3 months', '3-12 months', '1yr+'])
plt.figure(figsize=(12, 6))
sns.boxplot(x='max_stay_group', y='price', data=Sydney1, showfliers=False)
plt.title("Price Distribution by Maximum Stay Allowed")
plt.xticks(rotation=45)
plt.show()
Sydney1.describe()
"""##Visualizations Univariate Analysis##
"""
# Get numerical columns (int and float)
num_cols = Sydney1.select_dtypes(include=['int64', 'float64']).columns
n_cols = len(num_cols)
# Calculate grid size dynamically
n_rows = (n_cols // 4) + (1 if n_cols % 4 != 0 else 0)  # 4 columns per row
plt.figure(figsize=(20, 5 * n_rows))  # Adjust height based on rows
for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, 4, i)
    sns.histplot(data=Sydney1, x=col, bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
# Visualization for categorical variables
# Identify categorical columns (excluding any list-type columns)
cat_cols = [col for col in Sydney1.select_dtypes(include=['object', 'category']).columns
            if not any(isinstance(x, list) for x in Sydney1[col].head())]
# Now create the visualization
plt.figure(figsize=(16, 20))
for i, col in enumerate(cat_cols[:8]):  # First 8 valid categorical columns
    plt.subplot(4, 2, i+1)
    # Convert to string if needed (safe handling)
    if Sydney1[col].apply(type).eq(list).any():
        plot_data = Sydney1[col].astype(str).value_counts().head(10)
    else:
        plot_data = Sydney1[col].value_counts().head(10)
    plot_data.plot(kind='bar')
    plt.title(f'Distribution of {col}', pad=10)
    plt.xticks(rotation=45)
    plt.xlabel('')

plt.tight_layout()
plt.show()
"""##Bivariate Analysis and Price Exploration##"""
"""Bivariate Analysis"""
# Calculate key percentiles for price
price_stats = Sydney1['price'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print("\n--- Price Percentiles ---")
print(price_stats)
# Visualize price distribution with outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=Sydney1['price'])
plt.title('Price Distribution with Outliers')
plt.show()
# Set style for better visuals
sns.set(style="whitegrid")
# Price Distribution by Room Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='room_type', y='price', data=Sydney1)
plt.title('Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.show()
# Price Distribution by Room Type (Log Scale)
plt.figure(figsize=(12, 6))
sns.boxplot(x='room_type', y='price', data=Sydney1)
plt.yscale('log')  # Log scale to better visualize wide price ranges
plt.title('Price Distribution by Room Type (Log Scale)')
plt.xlabel('Room Type')
plt.ylabel('Price ($) - Log Scale')
plt.xticks(rotation=45)
plt.show()
# Price Distribution by Top Property Types
# Get top N property types by count (adjust number as needed)
top_properties = Sydney1['property_type'].value_counts().nlargest(10).index
plt.figure(figsize=(14, 6))
sns.boxplot(
    x='property_type',
    y='price',
    data=Sydney1[Sydney1['property_type'].isin(top_properties)],
    order=top_properties
)
plt.title('Price Distribution by Top Property Types')
plt.xlabel('Property Type')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.show()
# Price Distribution by Top Property Types (Log Scale)
plt.figure(figsize=(14, 6))
sns.boxplot(
    x='property_type',
    y='price',
    data=Sydney1[Sydney1['property_type'].isin(top_properties)],
    order=top_properties
)
plt.yscale('log')
plt.title('Price Distribution by Top Property Types (Log Scale)')
plt.xlabel('Property Type')
plt.ylabel('Price ($) - Log Scale')
plt.xticks(rotation=45)
plt.show()
# Average Price by Room and Property Type (Heatmap)
# Create a cross-tabulation of average prices
cross_tab = pd.pivot_table(
    Sydney1[Sydney1['property_type'].isin(top_properties)],
    values='price',
    index='property_type',
    columns='room_type',
    aggfunc=np.median
)
plt.figure(figsize=(12, 8))
sns.heatmap(cross_tab, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('Median Price by Property Type and Room Type ($)')
plt.xlabel('Room Type')
plt.ylabel('Property Type')
plt.show()
# Create the summary table
price_summary = Sydney1.groupby(['year', 'room_type']).agg(
    Listings_Count=('id', 'count'),
    Avg_Price=('price', 'mean'),
    Median_Price=('price', 'median')
).reset_index()
# Pivot to get room types as columns
pivoted_price = price_summary.pivot_table(
    index='year',
    columns='room_type',
    values=['Avg_Price', 'Median_Price'],
    fill_value=0
)
# Flatten multi-index columns
pivoted_price.columns = [f"{col[1]}_{col[0]}" for col in pivoted_price.columns]
pivoted_price.reset_index(inplace=True)
# Format the table for display
formatted_price_table = pivoted_price.copy()
formatted_price_table['Year'] = formatted_price_table['year']
formatted_price_table = formatted_price_table[[
    'Year',
    'Entire home/apt_Avg_Price', 'Private room_Avg_Price', 'Shared room_Avg_Price',
    'Entire home/apt_Median_Price', 'Private room_Median_Price', 'Shared room_Median_Price'
]]
# Rename columns to match your requested format
formatted_price_table.columns = [
    'Year',
    'Avg Price (Entire Home)', 'Avg Price (Private Room)', 'Avg Price (Shared Room)',
    'Median Price (Entire Home)', 'Median Price (Private Room)', 'Median Price (Shared Room)'
]
# Apply currency formatting
for col in formatted_price_table.columns[1:]:
    formatted_price_table[col] = formatted_price_table[col].apply(lambda x: f"${x:,.2f}")
print("\n--- Formatted Price Table by Year and Room Type (Avg/Median) ---")
display(formatted_price_table)
# Create summary statistics
price_stats_detailed = Sydney1.groupby(['year', 'room_type'])['price'].agg(
    Listings_Count='count',
    Min_Price='min',
    Max_Price='max',
    Avg_Price='mean',
    Median_Price='median'
).reset_index()
# Pivot to get room types as columns
pivoted_stats = price_stats_detailed.pivot_table(
    index='year',
    columns='room_type',
    values=['Min_Price', 'Max_Price', 'Avg_Price', 'Median_Price'],
    fill_value=0
)
# Flatten multi-index columns
pivoted_stats.columns = [f"{col[1]}_{col[0]}" for col in pivoted_stats.columns]
pivoted_stats.reset_index(inplace=True)

# Create final formatted table
final_price_table_detailed = pivoted_stats[[
    'year',
    'Entire home/apt_Min_Price', 'Private room_Min_Price', 'Shared room_Min_Price',
    'Entire home/apt_Max_Price', 'Private room_Max_Price', 'Shared room_Max_Price',
    'Entire home/apt_Avg_Price', 'Private room_Avg_Price', 'Shared room_Avg_Price',
    'Entire home/apt_Median_Price', 'Private room_Median_Price', 'Shared room_Median_Price'
]]
# Rename columns
final_price_table_detailed.columns = [
    'Year',
    'Min (Entire)', 'Min (Private)', 'Min (Shared)',
    'Max (Entire)', 'Max (Private)', 'Max (Shared)',
    'Avg (Entire)', 'Avg (Private)', 'Avg (Shared)',
    'Median (Entire)', 'Median (Private)', 'Median (Shared)'
]
# Apply currency formatting
price_cols = final_price_table_detailed.columns[1:]
final_price_table_detailed[price_cols] = final_price_table_detailed[price_cols].applymap(lambda x: f"${x:,.2f}")
print("\n--- Detailed Price Table by Year and Room Type (Min/Max/Avg/Median) ---")
display(final_price_table_detailed)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Beds
sns.histplot(Sydney1['beds'].dropna(), bins=30, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Beds')
axes[0].set_xlabel('Number of Beds')
axes[0].set_ylabel('Count')
# Bedrooms
sns.histplot(Sydney1['bedrooms'].dropna(), bins=30, ax=axes[1], color='salmon')
axes[1].set_title('Distribution of Bedrooms')
axes[1].set_xlabel('Number of Bedrooms')
# Bathrooms
sns.histplot(Sydney1['bathrooms'].dropna(), bins=30, ax=axes[2], color='lightgreen')
axes[2].set_title('Distribution of Bathrooms')
axes[2].set_xlabel('Number of Bathrooms')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(data=Sydney1[['beds', 'bedrooms', 'bathrooms']], palette="Set2")
plt.title('Beds, Bedrooms, and Bathrooms Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Numeric features vs price
num_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'number_of_reviews', 'review_scores_rating']
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=Sydney1[col], y=Sydney1['price'], alpha=0.5)
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()
# Correlation Matrix
print("\n--- Correlation Matrix of Numerical Features ---")
numerical_cols = Sydney1.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = Sydney1[numerical_cols].corr()
# Plot heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix (Numerical Features)')
plt.show()
price_corr = corr_matrix['price'].sort_values(ascending=False)
print("\n--- Top features correlated with price ---")
print(price_corr)
# Price Distribution (Original vs Log Scale)
plt.figure(figsize=(12, 6))
sns.histplot(Sydney1['price'], bins=50, kde=True)
plt.title('Price Distribution (Original Scale)')
plt.xlabel('Price ($)')
plt.show()
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(Sydney1['price']), bins=50, kde=True)
plt.title('Price Distribution (Log Scale)')
plt.xlabel('log(Price)')
plt.show()
# Price by category (room_type)
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='room_type',
    y='price',
    data=Sydney1
)
plt.title('Price Distribution by Room Type')
plt.xticks(rotation=45)
plt.show()
# Price by category (property_type)
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='property_type',
    y='price',
    data=Sydney1
)
plt.title('Price Distribution by Property Type')
plt.xticks(rotation=45)
plt.show()
# Price by category (host_is_superhost)
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='host_is_superhost',
    y='price',
    data=Sydney1
)
plt.title('Price Distribution by host_is_superhost')
plt.xticks(rotation=45)
plt.show()
# Price by category (host_neighbourhood) - showing top 10 neighbourhoods by listing count
top_neighbourhoods = Sydney1['host_neighbourhood'].value_counts().nlargest(10).index
plt.figure(figsize=(15, 7))
sns.boxplot(
    x='host_neighbourhood',
    y='price',
    data=Sydney1[Sydney1['host_neighbourhood'].isin(top_neighbourhoods)],
    order=top_neighbourhoods
)
plt.title('Price Distribution by Top 10 Neighbourhoods')
plt.xticks(rotation=90)
plt.show()
"""##Data Quality Checks and Further Refinements ##"""
# Host Since Analysis
print("\n=== Host Since ===")
print("Date Range:",
      Sydney1['host_since'].min().strftime('%Y-%m-%d'), "to",
      Sydney1['host_since'].max().strftime('%Y-%m-%d'))
print("Median Host Tenure:",
      (datetime.now() - Sydney1['host_since'].median()).days, "days")
# Last Review Analysis (with missing value handling)
print("\n=== Last Review ===")
if Sydney1['last_review'].notna().any():
    print("Date Range (non-null):",
          Sydney1['last_review'].min().strftime('%Y-%m-%d'), "to",
          Sydney1['last_review'].max().strftime('%Y-%m-%d'))
else:
    print("No valid review dates found")
print("Null Values:", Sydney1['last_review'].isna().sum())
# Review Date Consistency Check
active_without_reviews = Sydney1[
    (Sydney1['number_of_reviews'] > 0) &
    (Sydney1['last_review'].isna())
]
print("\n=== Data Quality Issues (Date) ===")
print(f"Listings with reviews but missing 'last_review' dates: {len(active_without_reviews)}")
print(f"Listings with future 'last_review' dates: {len(Sydney1[Sydney1['last_review'] > datetime.now()])}")
# Host Response Rate and Acceptance Rate Analysis
print("\n--- Host Response Rate Info ---")
print(Sydney1['host_response_rate'].info())
print("\n--- Host Response Rate Head ---")
print(Sydney1['host_response_rate'].head())
print("\n--- Host Response Rate Description ---")
print(Sydney1['host_response_rate'].describe())
print("\n--- Host Acceptance Rate Info ---")
print(Sydney1['host_acceptance_rate'].info())
print("\n--- Host Acceptance Rate Head ---")
print(Sydney1['host_acceptance_rate'].head())
print("\n--- Host Acceptance Rate Info (re-check) ---")
print(Sydney1['host_acceptance_rate'].info())
# Convert percentage strings to numeric (re-run in case previous imputation didn't handle it fully)
def convert_percentage(rate):
    if pd.isna(rate):
        return np.nan
    # Check if it's already a float (from previous imputation)
    if isinstance(rate, (float, int)):
        return float(rate)
    return float(str(rate).strip('%'))/100
Sydney1['host_response_rate_num'] = Sydney1['host_response_rate'].apply(convert_percentage)
Sydney1['host_acceptance_rate_num'] = Sydney1['host_acceptance_rate'].apply(convert_percentage)
# Check missingness correlation
missing_data_rates = Sydney1[['host_response_rate_num', 'host_acceptance_rate_num']].isnull()
print("\n--- Missing Value Correlation (Response/Acceptance Rates) ---")
print(missing_data_rates.corr())
# Compare superhost vs non-superhost
print("\n--- Missing rates by superhost status ---")
print(missing_data_rates.groupby(Sydney1['host_is_superhost']).mean())
# Comparative analysis with review scores
review_cols = ['review_scores_rating', 'review_scores_communication']
for rate in ['host_response_rate_num', 'host_acceptance_rate_num']:
    print(f"\n--- Correlation between {rate} and review scores ---")
    print(Sydney1[[rate] + review_cols].corr())

performance_metrics = ['number_of_reviews', 'price', 'availability_365']
for rate in ['host_response_rate_num', 'host_acceptance_rate_num']:
    plt.figure(figsize=(10,4))
    for i, metric in enumerate(performance_metrics):
        plt.subplot(1,3,i+1)
        sns.scatterplot(x=rate, y=metric, data=Sydney1, alpha=0.3)
        plt.title(f'{rate} vs {metric}')
    plt.tight_layout()
    plt.show()
# Value range validation
# Check for impossible values
issues = {
    'Negative Prices': len(Sydney1[Sydney1['price'] <= 0]),
    'Zero Accommodates': len(Sydney1[Sydney1['accommodates'] == 0]),
    'Availability >365': len(Sydney1[Sydney1['availability_365'] > 365]),
    'Bedrooms >20': len(Sydney1[Sydney1['bedrooms'] > 20]),
    'Bathrooms >10': len(Sydney1[Sydney1['bathrooms'] > 10])
}
print("\n--- Data Quality Issues (Value Ranges) ---")
for issue, count in issues.items():
    print(f"{issue}: {count}")
# Get listings with more than 10 bathrooms
luxury_bathrooms = Sydney1[Sydney1['bathrooms'] > 10]
print("\n--- Listings with more than 10 bathrooms ---")
display(luxury_bathrooms[[
    'id', 'name', 'host_name', 'property_type',
    'bathrooms', 'bedrooms', 'accommodates', 'price',
    'latitude', 'longitude'
]].sort_values('bathrooms', ascending=False))
# Review score validation
# All review scores should be between 1-10
review_cols_num = [c for c in Sydney1.columns if c.startswith('review_scores_') and c != 'reviews_per_month'] # Exclude reviews_per_month
invalid_reviews = Sydney1[(Sydney1[review_cols_num] < 1).any(axis=1) |
                             (Sydney1[review_cols_num] > 10).any(axis=1)]
print("\n--- Entries with invalid review scores (outside 1-10 range) ---")
print(f"Number of listings with invalid review scores: {len(invalid_reviews)}")
print("\nReview Score Columns Checked:", review_cols_num)
# Logical relationship (Capacity Consistency)
capacity_issues_df = Sydney1[
    (Sydney1['beds'] > Sydney1['accommodates']) |
    (Sydney1['bedrooms'] > Sydney1['accommodates']) |
    (Sydney1['bathrooms'] > Sydney1['accommodates'])
]
print("\n--- Properties with initial capacity inconsistencies ---")
print(f"Total properties with capacity inconsistencies: {len(capacity_issues_df)}")
# Breakdown of inconsistency types
print(f"Beds > Accommodates: {len(capacity_issues_df[capacity_issues_df['beds'] > capacity_issues_df['accommodates']])}")
print(f"Bedrooms > Accommodates: {len(capacity_issues_df[capacity_issues_df['bedrooms'] > capacity_issues_df['accommodates']])}")
print(f"Bathrooms > Accommodates: {len(capacity_issues_df[capacity_issues_df['bathrooms'] > capacity_issues_df['accommodates']])}")
def correct_capacity(df):
    """Corrects capacity inconsistencies where beds/bedrooms/bathrooms exceed accommodates."""
    original = df.copy()
    # Apply corrections for each column
    for col in ['beds', 'bedrooms', 'bathrooms']:
        # Only correct if the value is greater than accommodates
        mask = (df[col].notna()) & (df['accommodates'].notna()) & (df[col] > df['accommodates'])
        df.loc[mask, col] = df.loc[mask, 'accommodates']
    # Report changes
    print("\n--- Capacity Correction Report ---")
    for col in ['beds', 'bedrooms', 'bathrooms']:
        changed = (original[col] != df[col]).sum()
        print(f"Corrected {changed} {col} values.")
    return df
Sydney1 = correct_capacity(Sydney1)
def validate_capacity(df):
    """Validates if capacity issues remain after correction."""
    issues = {
        'beds': len(df[df['beds'] > df['accommodates']]),
        'bedrooms': len(df[df['bedrooms'] > df['accommodates']]),
        'bathrooms': len(df[df['bathrooms'] > df['accommodates']])
    }
    if any(issues.values()):
        print("\n--- Remaining issues after correction ---")
        for k, v in issues.items():
            print(f"{k}: {v}")
    else:
        print("\n--- All capacity constraints validated successfully! ---")
validate_capacity(Sydney1)
"""##Initial Data Exploration and Preprocessing##"""
print("\n--- Host Neighbourhood Information ---")
print(Sydney1['host_neighbourhood'].info())
print(Sydney1['host_neighbourhood'].head())
print(Sydney1['host_neighbourhood'].describe())
print("\n--- Neighbourhood Information ---")
print(Sydney1['neighbourhood_cleansed'].info())
print(Sydney1['neighbourhood_cleansed'].head())
print(Sydney1['neighbourhood_cleansed'].describe())
# Check missingness patterns
print("\n--- Missing Value Correlation Matrix ---")
missing_corr = Sydney1.isnull().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(missing_corr, cmap='coolwarm', center=0)
plt.title('Missing Value Correlation Matrix')
plt.show()
# Price comparison for top 5 neighborhoods vs Unknown
plt.figure(figsize=(10, 6))
sns.boxplot(x='host_neighbourhood', y='price',
            data=Sydney1[Sydney1['host_neighbourhood'].isin(
                ['Unknown'] + list(Sydney1['host_neighbourhood'].value_counts().index[:5]))])
plt.xticks(rotation=45)
plt.title('Price Distribution by Host Neighborhood (Top 5 vs Unknown)')
plt.show()
# Superhost rate comparison
print("\n--- Superhost Rates by Host Neighbourhood ---")
superhost_rates = Sydney1.groupby('host_neighbourhood')['host_is_superhost'].mean().sort_values(ascending=False)
print(superhost_rates.head(10))
# T-test for price differences between known and unknown host neighborhoods
known_price = Sydney1[Sydney1['host_neighbourhood'] != 'Unknown']['price']
unknown_price = Sydney1[Sydney1['host_neighbourhood'] == 'Unknown']['price']
t_stat, p_value = ttest_ind(known_price, unknown_price, equal_var=False)
print(f"\n--- T-test p-value for Price (Known vs Unknown Host Neighbourhood): {p_value:.4f} ---")
# Cross-tab with property type
print("\n--- Cross-tab: Host Neighbourhood vs Property Type ---")
print(pd.crosstab(Sydney1['host_neighbourhood'],
                  Sydney1['property_type'],
                  normalize='index').style.background_gradient(cmap='Blues'))
# Geographic plot of known vs unknown
known = Sydney1[Sydney1['host_neighbourhood'] != 'Unknown']
unknown = Sydney1[Sydney1['host_neighbourhood'] == 'Unknown']
plt.figure(figsize=(12, 8))
plt.scatter(known['longitude'], known['latitude'], alpha=0.3, label='Known')
plt.scatter(unknown['longitude'], unknown['latitude'], alpha=0.3, c='red', label='Unknown')
plt.legend()
plt.title('Geographic Distribution of Missing Host Neighborhoods')
plt.show()
# Temporal analysis (if host_since available)
Sydney1['host_since'] = pd.to_datetime(Sydney1['host_since'])
Sydney1['host_tenure'] = (pd.to_datetime('today') - Sydney1['host_since']).dt.days
sns.kdeplot(data=Sydney1, x='host_tenure', hue='host_neighbourhood', common_norm=False)
plt.title('Host Tenure Distribution by Neighborhood Status')
plt.show()
# Missingness analysis for neighbourhood_cleansed
print("\n=== Missing Value Analysis for neighbourhood_cleansed ===")
# 1. Basic missing value stats
missing_count = Sydney1['neighbourhood_cleansed'].isna().sum()
unknown_count = (Sydney1['neighbourhood_cleansed'] == 'Unknown').sum()
total_records = len(Sydney1)
print(f"Missing values: {missing_count} ({missing_count/total_records*100:.2f}%)")
print(f"'Unknown' values: {unknown_count} ({unknown_count/total_records*100:.2f}%)")
# 2. Price comparison for top 5 neighborhoods vs Missing/Unknown
plt.figure(figsize=(12, 6))
top_neighbourhoods = Sydney1['neighbourhood_cleansed'].value_counts().index[:5]
# Create comparison groups
comparison_groups = list(top_neighbourhoods) + ['Unknown']
if missing_count > 0:
    comparison_groups.append('Missing')
# Prepare data for visualization
plot_data = Sydney1.copy()
plot_data['neighbourhood_group'] = np.where(
    plot_data['neighbourhood_cleansed'].isna(),
    'Missing',
    plot_data['neighbourhood_cleansed']
)
plot_data = plot_data[plot_data['neighbourhood_group'].isin(comparison_groups)]
sns.boxplot(
    x='neighbourhood_group',
    y='price',
    data=plot_data,
    order=comparison_groups
)
plt.xticks(rotation=45)
plt.title('Price Distribution by Neighbourhood Status (Top 5 vs Unknown/Missing)')
plt.show()
# 3. Superhost rate comparison
print("\n--- Superhost Rates by Neighbourhood ---")
neighbourhood_stats = Sydney1.groupby('neighbourhood_cleansed').agg(
    superhost_rate=('host_is_superhost', 'mean'),
    avg_price=('price', 'mean'),
    count=('id', 'count')
).sort_values('count', ascending=False)
print(neighbourhood_stats.head(10))
# 4. T-test for price differences
known_price = Sydney1[~Sydney1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]['price']
unknown_price = Sydney1[Sydney1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]['price']

t_stat, p_value = ttest_ind(known_price, unknown_price, equal_var=False)
print(f"\nT-test p-value for Price (Known vs Unknown/Missing): {p_value:.4f}")
# 5. Cross-tab with property type
print("\n--- Property Type Distribution ---")
print(pd.crosstab(
    Sydney1['neighbourhood_cleansed'],
    Sydney1['property_type'],
    margins=True
).style.background_gradient(cmap='Blues'))
# 6. Geographic plot
plt.figure(figsize=(12, 8))
known = Sydney1[~Sydney1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]
unknown = Sydney1[Sydney1['neighbourhood_cleansed'].isin(['Unknown', np.nan])]
plt.scatter(known['longitude'], known['latitude'], alpha=0.3, label='Known')
plt.scatter(unknown['longitude'], unknown['latitude'], alpha=0.5, c='red', label='Unknown/Missing')
plt.legend()
plt.title('Geographic Distribution of Neighbourhood Status')
plt.show()
# 7. Temporal analysis (only if host_since exists)
if 'host_since' in Sydney1.columns:
    # Create host_tenure column if it doesn't exist
    if 'host_tenure' not in Sydney1.columns:
        Sydney1['host_tenure'] = (pd.to_datetime('today') - pd.to_datetime(Sydney1['host_since'])).dt.days
    plt.figure(figsize=(10, 6))
    # Create simplified groups for visualization
    plot_data = Sydney1.copy()
    plot_data['neighbourhood_status'] = np.where(
        plot_data['neighbourhood_cleansed'].isna(),
        'Missing',
        np.where(
            plot_data['neighbourhood_cleansed'] == 'Unknown',
            'Unknown',
            'Known'
        )
    )
    sns.kdeplot(
        data=plot_data,
        x='host_tenure',
        hue='neighbourhood_status',
        hue_order=['Known', 'Unknown', 'Missing'],
        common_norm=False
    )
    plt.title('Host Tenure by Neighbourhood Status')
    plt.xlabel('Host Tenure (days)')
    plt.show()
else:
    print("\nNote: 'host_since' column not available for temporal analysis")
Sydney1.to_csv('sydney_airbnb_listings.csv', index=False)
print("Saved as CSV: sydney_airbnb_listings.csv")
"""##VIF##
VIF Variance Inflation Factor for Multicollinearity
"""
#Vif Check
# Assume your DataFrame is called df
# Step 1: Drop irrelevant columns
exclude_cols = [
    'id', 'name', 'host_id', 'host_name', 'host_since', 'neighbourhood', 'host_neighbourhood', 'neighbourhood_cleansed',
    'property_type', 'room_type', 'amenities', 'last_review', 'host_since', 'first_review' # categorical or object columns
]
# Also drop boolean and datetime columns for VIF calculation
exclude_types = ['bool', 'datetime64[ns]', 'category', 'object']
# Filter to only numeric columns not in exclude_cols
vif_Sydney = Sydney1.drop(columns=exclude_cols)
vif_Sydney = vif_Sydney.select_dtypes(exclude=exclude_types)
# Step 2: Add constant for VIF calculation
X = add_constant(vif_Sydney)
# Step 3: Calculate VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# Step 4: Sort and display
print(vif_data.sort_values(by="VIF", ascending=False))
"""##Correlation##"""
# Calculate Spearman correlation matrix
spearman_corr = Sydney1.select_dtypes(include=['int64','float64']).corr(method='spearman')
# Plot heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Spearman Correlation Matrix')
plt.show()
# Get top features correlated with price
price_corr_spearman = spearman_corr['price'].sort_values(key=abs, ascending=False)
print("\nTop features correlated with price (Spearman):")
print(price_corr_spearman.head(15))
"""##Feature selection##"""
from sklearn.ensemble import RandomForestRegressor
# Prepare data
X = Sydney1.select_dtypes(include=['int64','float64']).drop(columns=['price'])
y = Sydney1['price']
# Fit model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Random Forest Feature Importance')
plt.show()
import xgboost as xgb
# Fit model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X, y)
# Get feature importance
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importance.head(15))
plt.title('XGBoost Feature Importance')
plt.show()
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Fit Lasso with cross-validation to find best alpha
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)
# Get coefficients
lasso_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=lasso_coef.head(15))
plt.title('Lasso Regression Feature Importance')
plt.show()
# Features kept by Lasso (non-zero coefficients)
selected_features = X.columns[lasso.coef_ != 0]
print("\nFeatures selected by Lasso:")
print(selected_features)
# Combine results from all methods
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Spearman': abs(spearman_corr['price'][X.columns]),
    'RF': rf.feature_importances_,
    'XGB': xgb_model.feature_importances_,
    'Lasso': abs(lasso.coef_)
})
# Normalize scores
feature_scores['RF'] = feature_scores['RF'] / feature_scores['RF'].max()
feature_scores['XGB'] = feature_scores['XGB'] / feature_scores['XGB'].max()
feature_scores['Lasso'] = feature_scores['Lasso'] / feature_scores['Lasso'].max()
# Calculate consensus score
feature_scores['Consensus'] = feature_scores[['Spearman', 'RF', 'XGB', 'Lasso']].mean(axis=1)
# Get top features
top_features = feature_scores.sort_values('Consensus', ascending=False).head(15)
# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Consensus', y='Feature', data=top_features)
plt.title('Consensus Feature Importance')
plt.show()
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (r2_score, mean_squared_error,
                           mean_absolute_error, mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.inspection import PartialDependenceDisplay
# ======================
# FEATURE SELECTION
# ======================
selected_features = [
    'bedrooms', 'longitude', 'latitude', 'bathrooms',
    'accommodates', 'review_scores_location', 'beds',
    'availability_365', 'host_total_listings_count',
    'reviews_per_month'
]
# Define a function to calculate all metrics
def calculate_metrics(model, X_train, y_train, X_test, y_test, cv=None):
    """Calculate all requested metrics for train and test sets"""
    # Train metrics
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'Train R2': r2_score(y_train, y_train_pred),
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Train MAPE': mean_absolute_percentage_error(y_train, y_train_pred)
    }
    # Test metrics
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'Test R2': r2_score(y_test, y_test_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred)
    }
    # Cross-validation metrics if cv is provided
    cv_metrics = {}
    if cv:
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        cv_metrics = {
            'CV R2 Mean': np.mean(cv_r2),
            'CV R2 Std': np.std(cv_r2),
            'CV R2 Min': np.min(cv_r2),
            'CV R2 Max': np.max(cv_r2)
        }
    return {**train_metrics, **test_metrics, **cv_metrics}
# Prepare data
X = Sydney1[selected_features]
y = Sydney1['price']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Initialize models (same as before)
models = {
    'OLS': LinearRegression(),
    'Lasso': Lasso(alpha=0.01),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'XGBoost': XGBRegressor(random_state=42)
}
# Set up cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Evaluate all models
results = []
for name, model in models.items():
    # Handle scaling for models that need it
    if name in ['OLS', 'Lasso', 'SVR', 'KNN']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    # Fit model
    model.fit(X_tr, y_train)
    # Calculate metrics
    metrics = calculate_metrics(model, X_tr, y_train, X_te, y_test, cv=cv)
    # Store results
    results.append({
        'Model': name,
        **metrics
    })
# Create results dataframe
results_df = pd.DataFrame(results)
# Reorder columns for better readability
column_order = [
    'Model',
    'Train R2', 'Test R2', 'CV R2 Mean', 'CV R2 Std',
    'Train MSE', 'Test MSE',
    'Train RMSE', 'Test RMSE',
    'Train MAE', 'Test MAE',
    'Train MAPE', 'Test MAPE',
    'CV R2 Min', 'CV R2 Max'
]
results_df = results_df[column_order]
# Display results sorted by Test R2
print(results_df.sort_values('Test R2', ascending=False).round(4))
# Save to CSV if needed
results_df.to_csv('model_performance_metrics.csv', index=False)
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
# Prepare data (scaled for KNN/OLS)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# =============================================
# 1. Gradient Boosting - In-Depth Analysis
# =============================================
gb = GradientBoostingRegressor(random_state=42)
# Parameter tuning
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
gb_cv = RandomizedSearchCV(gb, gb_param_grid, n_iter=20, cv=5,
                          scoring='r2', random_state=42)
gb_cv.fit(X_train, y_train)
print("=== Gradient Boosting ===")
print("Best Parameters:", gb_cv.best_params_)
print(f"Best CV R2: {gb_cv.best_score_:.4f}")
# Feature Importance
gb_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': gb_cv.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=gb_importance.head(10))
plt.title("Gradient Boosting Feature Importance (Top 10)")
plt.show()
# Partial Dependence Plots
print("\nPartial Dependence Plots (Top 3 Features):")
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    gb_cv.best_estimator_,
    X_train,
    features=[0, 1, 2],
    feature_names=selected_features,
    ax=ax
)
plt.suptitle("Gradient Boosting Partial Dependence")
plt.tight_layout()
plt.show()
# =============================================
# 2. KNN - In-Depth Analysis
# =============================================
knn = KNeighborsRegressor()
# Parameter tuning
knn_param_grid = {
    'n_neighbors': range(3, 21),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1: manhattan, 2: euclidean
}
knn_cv = GridSearchCV(knn, knn_param_grid, cv=5, scoring='r2')
knn_cv.fit(X_train_scaled, y_train)
print("\n=== KNN ===")
print("Best Parameters:", knn_cv.best_params_)
print(f"Best CV R2: {knn_cv.best_score_:.4f}")
# Distance Analysis
distances, indices = knn_cv.best_estimator_.kneighbors(X_train_scaled[:5])  # First 5 samples
print("\nNeighbor distances for first 5 samples:")
print(distances)
# =============================================
# 3. OLS - In-Depth Analysis with Statsmodels
# =============================================
# Add constant for statsmodels (intercept term)
X_train_const = sm.add_constant(X_train_scaled)
X_test_const = sm.add_constant(X_test_scaled)
# Fit OLS model
ols = sm.OLS(y_train, X_train_const)
ols_results = ols.fit()
print("\n=== OLS Regression Summary ===")
print(ols_results.summary())

# Create coefficient plot
coef_df = pd.DataFrame({
    'Feature': ['Intercept'] + selected_features,
    'Coefficient': ols_results.params,
    'P-value': ols_results.pvalues
}).sort_values('Coefficient', key=abs, ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(15))
plt.title("OLS Coefficient Magnitudes (Top 15)")
plt.axvline(0, color='k', linestyle='--')
plt.show()
# Random Forest analysis
rf = RandomForestRegressor(random_state=42, oob_score=True)
rf.fit(X_train, y_train)
# OOB score
print("OOB R2:", rf.oob_score_)
# Tune parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_depth': [5, 10, 20, None]
}
rf_cv = RandomizedSearchCV(rf, param_grid, n_iter=20, cv=5, scoring='r2', random_state=42)
rf_cv.fit(X_train, y_train)
print("Best Random Forest Parameters:", rf_cv.best_params_)
print("Best CV R2:", rf_cv.best_score_)
# Feature importance
rf_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_cv.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=rf_importance)
plt.title("Random Forest Feature Importance")
plt.show()
# XGBoost analysis
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)

# Tune parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_cv = RandomizedSearchCV(xgb, param_grid, n_iter=20, cv=5, scoring='r2', random_state=42)
xgb_cv.fit(X_train, y_train)
print("Best XGBoost Parameters:", xgb_cv.best_params_)
print("Best CV R2:", xgb_cv.best_score_)
# Feature importance
xgb_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': xgb_cv.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance)
plt.title("XGBoost Feature Importance")
plt.show()
# Partial dependence plots
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(
    xgb_cv.best_estimator_,
    X_train,
    features=[0, 1, 2],  # Top 3 features
    feature_names=selected_features,
    ax=ax
)
plt.suptitle("Partial Dependence Plots")
plt.tight_layout()
plt.show()
from sklearn.metrics import (r2_score, mean_squared_error,
                           mean_absolute_error, mean_absolute_percentage_error)
from sklearn.model_selection import cross_val_score, KFold
# Initialize models with their best parameters from tuning
tuned_models = {
    'Random Forest': rf_cv.best_estimator_,
    'XGBoost': xgb_cv.best_estimator_,
    'Gradient Boosting': gb_cv.best_estimator_,
    'KNN': knn_cv.best_estimator_,
    'OLS': LinearRegression()  # Using sklearn's implementation for consistency
}
# Set up evaluation framework
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a model and return comprehensive metrics"""
    # Handle scaling for models that need it
    if model_name in ['KNN', 'OLS']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    # Fit model
    model.fit(X_tr, y_train)
    # Predictions
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train R2': r2_score(y_train, y_train_pred),
        'Test R2': r2_score(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train MAPE': mean_absolute_percentage_error(y_train, y_train_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred)
    }
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='r2')
    metrics.update({
        'CV R2 Mean': np.mean(cv_scores),
        'CV R2 Std': np.std(cv_scores)
    })
    return metrics

# Evaluate all models
results = []
for name, model in tuned_models.items():
    results.append(evaluate_model(model, X_train, y_train, X_test, y_test, name))
# Create results dataframe
results_df = pd.DataFrame(results)
# Reorder columns for better readability
column_order = [
    'Model',
    'Train R2', 'Test R2', 'CV R2 Mean', 'CV R2 Std',
    'Train RMSE', 'Test RMSE',
    'Train MAE', 'Test MAE',
    'Train MAPE', 'Test MAPE'
]
results_df = results_df[column_order]
# Display results sorted by Test R2
print("=== Model Performance Comparison ===")
print(results_df.sort_values('Test R2', ascending=False).round(4))
# =============================================
# Visualization 1: Model Comparison Bar Plot
# =============================================
plt.figure(figsize=(14, 8))
metrics_to_plot = ['Test R2', 'Test RMSE', 'Test MAE']
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(1, 3, i)
    sns.barplot(x=metric, y='Model', data=results_df.sort_values(metric, ascending=i>1))
    plt.title(metric)
    if metric == 'Test R2':
        plt.xlim(0, 1)  # R2 ranges 0-1
plt.tight_layout()
plt.suptitle("Model Performance Comparison (Test Set Metrics)", y=1.02)
plt.show()
# =============================================
# Visualization 2: Actual vs Predicted Values
# =============================================
plt.figure(figsize=(15, 10))
for i, (name, model) in enumerate(tuned_models.items(), 1):
    # Handle scaling
    if name in ['KNN', 'OLS']:
        X_te = X_test_scaled
    else:
       X_te = X_test
    y_pred = model.predict(X_te)
    plt.subplot(2, 3, i)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name}\nTest R2: {results_df[results_df['Model']==name]['Test R2'].values[0]:.3f}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    # Add histogram of residuals
    if i == len(tuned_models):  # Last subplot
        plt.subplot(2, 3, i+1)
        residuals = np.concatenate([y_test - model.predict(X_te)
                                  for model in tuned_models.values()])
        model_names = np.repeat(list(tuned_models.keys()), len(y_test))
        sns.boxplot(x=residuals, y=model_names)
        plt.title("Residual Distribution Comparison")
        plt.xlabel("Residuals")
        plt.ylabel("Model")
plt.tight_layout()
plt.show()
# =============================================
# Visualization 3: Metric Comparison Radar Chart
# =============================================
from math import pi
# Normalize metrics for radar chart (0-1 scale)
def normalize_series(s):
    if s.name in ['Test R2', 'CV R2 Mean']:  # Higher is better
        return (s - s.min()) / (s.max() - s.min())
    else:  # RMSE, MAE, MAPE - lower is better
        return 1 - ((s - s.min()) / (s.max() - s.min()))
radar_df = results_df.set_index('Model')[['Test R2', 'Test RMSE', 'Test MAE', 'Test MAPE', 'CV R2 Mean']]
radar_df = radar_df.apply(normalize_series, axis=0)
# Plot radar chart
categories = list(radar_df.columns)
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
for idx, row in radar_df.iterrows():
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=idx)
    ax.fill(angles, values, alpha=0.1)
plt.title("Model Performance Radar Chart\n(Normalized metrics, 1=best)", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()
# =============================================
# Additional: Time Efficiency Comparison
# =============================================
import time
timing_results = []
for name, model in tuned_models.items():
    # Handle scaling
    if name in ['KNN', 'OLS']:
        X_tr = X_train_scaled
    else:
        X_tr = X_train
    # Training time
    start = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - start
    # Prediction time (on test set)
    start = time.time()
    model.predict(X_tr[:100])  # Predict on 100 samples
    pred_time = time.time() - start
    timing_results.append({
        'Model': name,
        'Training Time (s)': train_time,
        'Prediction Time/100 (s)': pred_time
    })
timing_df = pd.DataFrame(timing_results)
print("\n=== Computational Efficiency ===")
print(timing_df.sort_values('Training Time (s)'))
# =============================================
# Enhanced Learning Curves & Residual Analysis
# =============================================
from sklearn.model_selection import learning_curve
def enhanced_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1)
    plt.figure(figsize=(12,6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='b', label="Training score")
    plt.fill_between(train_sizes,
                    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                    alpha=0.1, color='b')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='r', label="CV score")
    plt.fill_between(train_sizes,
                    np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                    np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                    alpha=0.1, color='r')

    plt.title(f"{model_name} Learning Curve\n(Shaded areas show ±1 std. dev.)")
    plt.xlabel("Training examples")
    plt.ylabel("R2 Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
def enhanced_residual_analysis(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    fig, ax = plt.subplots(1, 3, figsize=(18,5))
    # Residuals vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[0])
    ax[0].axhline(y=0, color='r', linestyle='--')
    ax[0].set_title(f"{model_name} Residuals vs Predicted")
    ax[0].set_xlabel("Predicted Values")
    ax[0].set_ylabel("Residuals")
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=ax[1])
    ax[1].set_title(f"{model_name} Residual Distribution")
    # Q-Q Plot
    sm.qqplot(residuals, line='s', ax=ax[2])
    ax[2].set_title(f"{model_name} Q-Q Plot")
    plt.tight_layout()
    plt.show()
    # Calculate residual statistics
    print(f"\n{model_name} Residual Analysis:")
    print(f"Mean residuals: {residuals.mean():.4f}")
    print(f"Residual std. dev.: {residuals.std():.4f}")
    print(f"Percentage of residuals within ±1 std. dev.: {(residuals.abs() <= residuals.std()).mean()*100:.2f}%")
# Run analyses for all five models
models_to_analyze = {
    'KNN': knn_cv.best_estimator_,
    'OLS': LinearRegression(),
    'Gradient Boosting': gb_cv.best_estimator_,
    'Random Forest': rf_cv.best_estimator_,
    'XGBoost': xgb_cv.best_estimator_
}
# Prepare scaled data for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add constant for OLS if using statsmodels
X_train_const = sm.add_constant(X_train_scaled)
X_test_const = sm.add_constant(X_test_scaled)
for name, model in models_to_analyze.items():
    print(f"\n{'='*40}\nAnalyzing: {name}\n{'='*40}")
    # Select proper input data
    if name in ['KNN', 'OLS']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    # Learning curve
    enhanced_learning_curve(model, X_tr, y_train, name)
    # Residual analysis
    enhanced_residual_analysis(model, X_tr, y_train, X_te, y_test, name)
    # For OLS, add VIF analysis
    if name == 'OLS':
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif = pd.DataFrame()
        vif["Feature"] = ['Intercept'] + selected_features
        vif["VIF"] = [variance_inflation_factor(X_train_const, i) for i in range(X_train_const.shape[1])]
        print("\nVariance Inflation Factors (VIF):")
        print(vif.sort_values('VIF', ascending=False))
from sklearn.model_selection import cross_validate
# Define KFold (consistent with your setup)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store CV results
cv_results = {}
for model_name, model in tuned_models.items():
    # Handle scaling for KNN/OLS
    X_data = X_train_scaled if model_name in ['KNN', 'OLS'] else X_train
    # Get CV scores (using cross_validate for flexibility)
    scores = cross_validate(
        model,
        X_data,
        y_train,
        cv=cv,
        scoring='r2',
        return_train_score=False,
        return_estimator=False
    )
    # Store individual fold scores and summary stats
    cv_results[model_name] = {
        'Fold Scores': scores['test_score'],
        'Mean CV R2': np.mean(scores['test_score']),
        'Std CV R2': np.std(scores['test_score'])
    }
# Convert to DataFrame for clean display
cv_df = pd.DataFrame({
    model: [
        cv_results[model]['Fold Scores'],
        cv_results[model]['Mean CV R2'],
        cv_results[model]['Std CV R2']
    ]
    for model in cv_results.keys()
}, index=['Fold Scores (R2)', 'Mean CV R2', 'Std CV R2']).T
print("=== K-Fold Cross-Validation Results ===")
display(cv_df)
# Optional: Save to CSV
cv_df.to_csv('kfold_cv_results.csv')
"""##Correlation with Price##"""
numerical_data_corr = Sydney1.select_dtypes(include=['int64', 'float64'])
corr_matrix = numerical_data_corr.corr()
print("\n--- Top Features Correlated with Price ---")
price_corr = corr_matrix[['price']].sort_values('price', ascending=False)
print(price_corr)
"""##Geographic and Relationship Visualizations##"""
# Geographic distribution by room type
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='longitude',
    y='latitude',
    hue='room_type',
    data=Sydney1,
    palette='viridis',
    alpha=0.6
)
plt.title('Geographic Distribution by Room Type')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# Getting the room details (Donut Chart)
roomdf = Sydney1.groupby('room_type').size()/Sydney1['room_type'].count()*100
labels = roomdf.index
values = roomdf.values
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
fig.update_layout(title_text='Distribution of Room Types')
fig.show()
# Catplot room type and price
plt.figure(figsize=(10,6))
sns.catplot(x="room_type", y="price", data=Sydney1);
plt.title('Price vs Room Type') # Add title to catplot
plt.ioff() # Turn off interactive mode, so plot doesn't show immediately if using it
plt.show() # Explicitly show the plot
# Word Cloud of Listing Names
text = " ".join(str(each) for each in Sydney1.name.dropna()) # Ensure no NaNs are passed
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200, background_color="white").generate(text)
plt.figure(figsize=(15,10)) # Adjust figure size for better display
# Display the generated image:
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.title('Word Cloud of Listing Names') # Add title
plt.show()
# Property Type Distribution
prop_type_counts = Sydney1['property_type'].value_counts(dropna=False)
print("\n--- Property Type Distribution ---")
print(prop_type_counts)
plt.figure(figsize=(12, 6))
sns.barplot(
    x=prop_type_counts.head(20).values,
    y=prop_type_counts.head(20).index,
    palette='viridis'
)
plt.title('Top 20 Property Types')
plt.xlabel('Number of Listings')
plt.ylabel('Property Type')
plt.tight_layout()
plt.show()
# Price Distribution by Top 20 Property Types (with outlier removal for visualization)
# Get top 20 property types by frequency
top_20_properties = Sydney1['property_type'].value_counts().nlargest(20).index
# Filter the data to include only top 20 property types and remove top 5% price outliers
filtered_data = Sydney1[
    (Sydney1['property_type'].isin(top_20_properties)) &
    (Sydney1['price'] <= Sydney1['price'].quantile(0.95))
]
# Create the plot
plt.figure(figsize=(12, 8))
sns.boxplot(
    y='property_type',
    x='price',
    data=filtered_data,
    palette='coolwarm',
    order=top_20_properties  # This ensures the property types are ordered by frequency
)
plt.title('Price Distribution by Top 20 Property Types (excluding top 5% price outliers)')
plt.ylabel('')
plt.xlabel('Price (AUD)')
plt.tight_layout()
plt.show()
# Room type breakdown by property type
print("\n--- Room Type Breakdown by Property Type ---")
display(pd.crosstab(
    Sydney1['property_type'],
    Sydney1['room_type'],
    margins=True
).sort_values('All', ascending=False))
# Geographic distribution by property type (Top 5)
if all(col in Sydney1.columns for col in ['latitude', 'longitude']):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='longitude',
        y='latitude',
        hue='property_type',
        data=Sydney1[Sydney1['property_type'].isin(prop_type_counts.head(5).index)],
        palette='tab10',
        alpha=0.6
    )
    plt.title('Geographic Distribution of Top 5 Property Types')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
    plt.tight_layout()
    plt.show()
# Statistical summary by property type
print("\n--- Price Statistics by Property Type ---")
print(
    Sydney1.groupby('property_type')['price']
    .describe(percentiles=[.25, .5, .75, .9])
    .sort_values('count', ascending=False))
# Room availability heatmap
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    x=Sydney1['longitude'],
    y=Sydney1['latitude'],
    c=Sydney1['availability_365'],  # Color by room_availability
    cmap='viridis',
    edgecolor='black',
    linewidth=0.5,
    alpha=0.7)
# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Room Availability (Days)')
# Add titles and labels
plt.title('Property Locations Colored by Room Availability')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Distribution of Availability
plt.figure(figsize=(10, 6))
sns.histplot(
    Sydney1['availability_365'],
    bins=30,
    kde=True,
    color='skyblue')
plt.title('Distribution of Availability (Days/Year)')
plt.xlabel('Days Available per Year')
plt.ylabel('Count')
plt.axvline(
    Sydney1['availability_365'].median(),
    color='red',
    linestyle='--',
    label=f'Median: {Sydney1["availability_365"].median():.0f} days')
plt.legend()
plt.show()
plt.figure(figsize=(12, 10))
# Simple geographic plot
sns.scatterplot(
    x='longitude',
    y='latitude',
    data=Sydney1,
    color='royalblue',  # Single color for all points
    alpha=0.5,
    s=10  # Point size)
# Add context
plt.title('Geographic Distribution of Airbnb Listings in Sydney', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
# Set proper aspect ratio for geographic data
ax = plt.gca()
ax.set_aspect(1.2)  # Adjust this value based on your coordinate range
# Add grid
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
import folium
from folium.plugins import MarkerCluster
# Calculate center point of all listings
center_lat = Sydney1['latitude'].mean()
center_lon = Sydney1['longitude'].mean()
# Create base map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
# Add marker clusters for better visualization of dense areas
marker_cluster = MarkerCluster().add_to(m)
# Add each listing to the map
for idx, row in Sydney1.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='royalblue',
        fill=True,
        fill_opacity=0.5,
        popup=f"Price: ${row['price']}",
        tooltip=row['neighbourhood_cleansed']
    ).add_to(marker_cluster)
# Add tile layer for better map styling
folium.TileLayer('cartodbpositron').add_to(m)  # Light theme
folium.TileLayer('openstreetmap').add_to(m)     # Standard OSM
folium.LayerControl().add_to(m)  # Allows toggling between map styles
# Add title
title_html = '''
    <h3 style="font-size:16px; margin-top:10px">
    <b>Geographic Distribution of Airbnb Listings in Sydney</b>
    </h3>'''
m.get_root().html.add_child(folium.Element(title_html))
# Display the map
display(m)
# Save to HTML file
m.save('sydney_airbnb_map.html')
"""##Geographic Feature Engineering ##"""
# Create base map centered on Sydney
m = folium.Map(location=[-33.8688, 151.2093], zoom_start=11)
# Create heatmap layer
# Ensure 'latitude' and 'longitude' are not NaN for heatmap
heat_data = Sydney1[['latitude', 'longitude']].dropna().values.tolist()
HeatMap(
    heat_data,
    radius=8,
    blur=10,
    gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'orange', 1.0: 'red'}
).add_to(m)
print("\n--- Interactive Heatmap of Listing Density in Sydney ---")
display(m)
# Distance to Sydney CBD (using Sydney Town Hall as reference point)
CBD_COORD = (-33.8738, 151.2069)
def calculate_distance(lat, lon, ref_point=CBD_COORD):
    """Calculate distance in kilometers from reference point."""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    return geodesic((lat, lon), ref_point).km
# Add distance to CBD column
Sydney1['distance_to_cbd'] = Sydney1.apply(
    lambda x: calculate_distance(x['latitude'], x['longitude']), axis=1
)
print("\n--- Distance to CBD Head ---")
print(Sydney1['distance_to_cbd'].head())
# Distance to major Sydney landmarks
SYDNEY_LANDMARKS = {
    'sydney_airport': (-33.9399, 151.1753),
    'central_station': (-33.8832, 151.2066),
    'university_of_sydney': (-33.8884, 151.1873),
    'opera_house': (-33.8568, 151.2153),
    'bondi_beach': (-33.8915, 151.2767),
    'manly_beach': (-33.8000, 151.2887),
    'darling_harbour': (-33.8747, 151.1989),
    'taronga_zoo': (-33.8466, 151.2419)}
# Calculate distances to each landmark
for name, coords in SYDNEY_LANDMARKS.items():
    Sydney1[f'distance_to_{name}'] = Sydney1.apply(
        lambda x: calculate_distance(x['latitude'], x['longitude'], coords), axis=1)
print("\n--- Distance to Landmarks Head ---")
print(Sydney1[['distance_to_sydney_airport', 'distance_to_opera_house', 'distance_to_bondi_beach']].head())
# Listing density features using BallTree
coords_for_tree = Sydney1[['latitude', 'longitude']].dropna()
if not coords_for_tree.empty:
    # Convert to radians for haversine metric
    coords_rad = np.deg2rad(coords_for_tree)
    tree = BallTree(coords_rad, metric='haversine')
    # Calculate listings within various radii
    for radius in [0.5, 1, 2, 5]:  # in kilometers
        rad_val = radius / 6371  # Convert km to radians
        listings = pd.Series(np.nan, index=Sydney1.index)
        listings.loc[coords_for_tree.index] = tree.query_radius(
            coords_rad, r=rad_val, count_only=True
        )
        Sydney1[f'listings_within_{radius}km'] = listings
        print(f"\nAdded listings within {radius}km radius")
    print("\n--- Listings Density Features Head ---")
    print(Sydney1[['listings_within_0.5km', 'listings_within_1km', 'listings_within_2km']].head())
else:
    print("Warning: No valid coordinates found for density calculations")
# Additional geographic features
def add_geographic_features(df):
    """Add derived geographic features"""
    # Coastal proximity flag (within 2km of major beaches)
    coastal_landmarks = ['bondi_beach', 'manly_beach']
    df['near_coast'] = df.apply(
        lambda x: any(x[f'distance_to_{lm}'] <= 2 for lm in coastal_landmarks
                      if not pd.isna(x[f'distance_to_{lm}'])), axis=1)
    # Transportation hub proximity
    df['near_transport_hub'] = df.apply(
        lambda x: x['distance_to_sydney_airport'] <= 10 or
                 x['distance_to_central_station'] <= 2, axis=1)
    # Tourist area flag
    tourist_landmarks = ['opera_house', 'darling_harbour', 'taronga_zoo']
    df['in_tourist_area'] = df.apply(
        lambda x: any(x[f'distance_to_{lm}'] <= 3 for lm in tourist_landmarks
                     if not pd.isna(x[f'distance_to_{lm}'])), axis=1)
    return df
Sydney1 = add_geographic_features(Sydney1)
print("\n--- Final DataFrame with Geographic Features ---")
print(Sydney1.info())
print("\n--- Sample of Geographic Features ---")
print(Sydney1[['distance_to_cbd', 'near_coast', 'near_transport_hub', 'in_tourist_area']].sample(5))
"""##Neighbourhood analysis##"""
# Create working dataset with only known neighborhoods
known_neighborhoods = Sydney1[Sydney1['host_neighbourhood'] != 'Unknown'].copy()
# Top neighborhoods by listings count
print("\n--- Top 10 Neighborhoods by Listings Count ---")
top_neighborhoods = known_neighborhoods['host_neighbourhood'].value_counts().nlargest(10)
print(top_neighborhoods)
# Neighborhood performance metrics
print("\n--- Neighborhood Performance Metrics ---")
neighborhood_stats = known_neighborhoods.groupby('host_neighbourhood').agg({
    'price': ['median', 'count'],
    'review_scores_rating': 'median',
    'host_is_superhost': 'mean',
    'number_of_reviews': 'median'
}).sort_values(('price', 'median'), ascending=False)
print(neighborhood_stats.head(10).to_string())
# ANOVA test for price differences among top 5 neighborhoods
top_5_neigh = known_neighborhoods['host_neighbourhood'].value_counts().nlargest(5).index
groups_anova = [known_neighborhoods[known_neighborhoods['host_neighbourhood']==n]['price']
                for n in top_5_neigh]
f_stat_anova, p_value_anova = f_oneway(*groups_anova)
print(f"\n--- ANOVA p-value for price differences (Top 5 Neighborhoods): {p_value_anova:.4f} ---")
"""##Demand Forecasting##"""
# Seasonal demand visualization
Sydney1['last_review'] = pd.to_datetime(Sydney1['last_review'])
Sydney1['month'] = Sydney1['last_review'].dt.month
monthly_demand = Sydney1.groupby('month')['number_of_reviews'].sum()
# Booking trends by month and season
Sydney1['review_month'] = Sydney1['last_review'].dt.month
Sydney1['review_year'] = Sydney1['last_review'].dt.year
seasonal_trends = Sydney1.groupby('review_month')['number_of_reviews'].sum().reset_index()
season_map = {
    1: 'Summer', 2: 'Summer', 3: 'Autumn',
    4: 'Autumn', 5: 'Autumn', 6: 'Winter',
    7: 'Winter', 8: 'Winter', 9: 'Spring',
    10: 'Spring', 11: 'Spring', 12: 'Summer'}
seasonal_trends['season'] = seasonal_trends['review_month'].map(season_map)
plt.figure(figsize=(10, 5))
sns.lineplot(x='review_month', y='number_of_reviews', data=seasonal_trends, marker='o')
plt.title("Airbnb Booking Trends by Month")
plt.xlabel("Month")
plt.ylabel("Total Reviews (Proxy for Bookings)")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()
print("\n--- Seasonal Performance ---")
seasonal_performance = seasonal_trends.groupby('season')['number_of_reviews'].sum().reset_index()
print(seasonal_performance)
"""##Host Performance##"""
# Superhost classifier feature importance
X_superhost = Sydney1[['host_response_rate_num', 'host_acceptance_rate_num', 'review_scores_rating']].dropna()
y_superhost = Sydney1.loc[X_superhost.index]['host_is_superhost']
clf = RandomForestClassifier(random_state=42)
clf.fit(X_superhost, y_superhost)
plt.figure(figsize=(8, 5))
pd.Series(clf.feature_importances_, index=X_superhost.columns).plot(kind='barh')
plt.title('Superhost Classifier Feature Importance')
plt.show()
"""##Geographic Analysis##"""
# Base map with CBD and top 50 properties by price
Syd_map = folium.Map(location=CBD_COORD, zoom_start=12)
folium.Marker(
    CBD_COORD,
    popup='Sydney CBD',
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(Syd_map)
for _, row in Sydney1.nlargest(50, 'price').iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color='blue',
        fill=True,
        popup=f"${row['price']}/night, {row['distance_to_cbd']:.1f}km to CBD"
    ).add_to(Syd_map)
print("\n--- Interactive Map with CBD and Top 50 Properties (displayed below) ---")
display(Syd_map)
"""##Operational Efficiency##"""
# Occupancy-Price tradeoff
plt.figure(figsize=(10, 6))
sns.scatterplot(x='availability_365', y='price',
                hue='room_type', data=Sydney1, alpha=0.6)
plt.title('Price vs Availability')
plt.show()
# Average Listing Price by Host Join Year
Sydney1['host_since_year'] = pd.to_datetime(Sydney1['host_since']).dt.year
plt.figure(figsize=(12,6))
sns.lineplot(
    x='host_since_year',
    y='price',
    data=Sydney1,
    estimator='mean',
    errorbar=None,
    marker='o')
plt.title('Average Listing Price by Host Join Year')
plt.xlabel('Year Host Joined')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()
# Average Listing Price by Host Join Month
Sydney1['host_since_month'] = pd.to_datetime(Sydney1['host_since']).dt.month
plt.figure(figsize=(12,6))
sns.lineplot(
    x='host_since_month',
    y='price',
    data=Sydney1,
    estimator='mean',
    errorbar=None,
    marker='o')
plt.title('Average Listing Price by Host Join Month')
plt.xlabel('Month Host Joined')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()
# Distribution of guest capacity
plt.figure(figsize=(10, 6))
sns.histplot(
    data=Sydney1,
    x='accommodates',
    bins=range(1, int(Sydney1['accommodates'].max())+2), # Ensure bins cover all integers
    kde=False,
    color='royalblue')
plt.title('Distribution of Guest Capacity')
plt.xlabel('Number of Guests Accommodated')
plt.ylabel('Number of Listings')
plt.xticks(range(1, int(Sydney1['accommodates'].max())+1))
plt.grid(axis='y', alpha=0.3)
plt.show()
Sydney1.info()
"""##Review Analysis##"""
# Distribution of number of reviews
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(Sydney1['number_of_reviews'], bins=50, kde=True)
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.subplot(1, 2, 2)
sns.boxplot(x=Sydney1['number_of_reviews'])
plt.title('Boxplot of Review Counts')
plt.tight_layout()
plt.show()
print("\n--- Statistical Summary of Number of Reviews ---")
print(Sydney1['number_of_reviews'].describe())

# Price vs Number of Reviews
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.regplot(x='number_of_reviews', y='price',
            data=Sydney1,
            scatter_kws={'alpha':0.3},
            line_kws={'color':'red'})
plt.title('Price vs Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Price ($)')
plt.subplot(1, 2, 2)
bins_reviews = [0, 5, 10, 20, 50, 100, 200, 500, Sydney1['number_of_reviews'].max()]
Sydney1['review_bins'] = pd.cut(Sydney1['number_of_reviews'], bins=bins_reviews, right=False)
sns.boxplot(x='review_bins', y='price', data=Sydney1)
plt.title('Price Distribution by Review Count Ranges')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Drill down into strong relationships (accommodates vs price)
sns.jointplot(x='accommodates', y='price', data=Sydney1, kind='reg')
plt.show()
# Check for nonlinear relationship (number_of_reviews vs price)
sns.lmplot(x='number_of_reviews', y='price',
           data=Sydney1,
           lowess=True,
           scatter_kws={'alpha':0.3})
plt.show()
"""##PCA Principal Component Analysis##"""
#######################################
# 1. PCA with Biplot Visualization
#######################################
# Select and prepare features
features = ['bedrooms', 'longitude', 'latitude', 'bathrooms',
    'accommodates', 'review_scores_location', 'beds',
    'availability_365', 'host_total_listings_count',
    'reviews_per_month'
]
X = Sydney1[features].fillna(Sydney1[features].median())
def run_pca_analysis(X, n_components=2, pipeline=None):
    """Perform PCA and create biplot with loadings, with option to use existing pipeline"""
    if pipeline is None:
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Fit PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
    else:
        # Use existing pipeline components
        scaler = pipeline.named_steps['scaler']
        pca = pipeline.named_steps['reducer']
        X_scaled = scaler.transform(X)
       principal_components = pca.transform(X_scaled)
    # Create results DataFrame
    pca_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC{i+1}' for i in range(n_components)])
    # Create biplot
    plt.figure(figsize=(15, 10))
    # Scatter plot of observations
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.6)
    # Plot feature loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    features = X.columns
    for i, feature in enumerate(features):
        plt.arrow(0, 0, loadings[i,0]*0.8, loadings[i,1]*0.8,
                 color='r', alpha=0.5, head_width=0.05)
        plt.text(loadings[i,0]*0.85, loadings[i,1]*0.85,
                feature, color='r', ha='center', va='center')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("PCA Biplot with Feature Loadings")
    plt.grid()
    # Scree plot
    plt.subplot(2, 2, 2)
    bars = plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
             np.cumsum(pca.explained_variance_ratio_),
             color='red', marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    # Print key variables
    print("\nKey Variables by PC:")
    for i in range(n_components):
        print(f"\nPC{i+1} Top Variables:")
        sorted_vars = sorted(zip(features, pca.components_[i]),
                         key=lambda x: abs(x[1]), reverse=True)[:5]
        for var, loading in sorted_vars:
            print(f"{var}: {loading:.3f}")
    # Component loadings table
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=features)
    print("\nComponent Loadings:")
    display(loadings_df)
    # Variance analysis
    print("\nVariance Analysis:")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", sum(pca.explained_variance_ratio_))
    return pca, pca_df, X_scaled
#######################################
# 2. K-means Clustering on PCA Scores
#######################################
def optimize_kmeans(data, max_k=10):
    """Determine optimal k using elbow and silhouette methods"""
    inertias = []
    silhouettes = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(data, kmeans.labels_))
    # Elbow plot
    kneedle = KneeLocator(range(2,max_k+1), inertias, curve='convex', direction='decreasing')
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(range(2,max_k+1), inertias, marker='o')
    if kneedle.elbow:
        plt.axvline(kneedle.elbow, color='r', linestyle='--')
    plt.title(f"Elbow Method (Suggested k={kneedle.elbow or 'Not clear'})")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    # Silhouette plot
    plt.subplot(1,2,2)
    plt.plot(range(2,max_k+1), silhouettes, marker='o')
    if len(silhouettes) > 0:
        optimal_k = np.argmax(silhouettes) + 2
        plt.axvline(optimal_k, color='r', linestyle='--')
    plt.title(f"Silhouette Method (Suggested k={optimal_k if 'optimal_k' in locals() else 'Not clear'})")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.show()
    return kneedle.elbow if kneedle.elbow else optimal_k
##############################
# 3. Hierarchical Clustering
def hierarchical_clustering(data, method='ward'):
    """Generate dendrogram for hierarchical clustering"""
    plt.figure(figsize=(12,6))
    linked = linkage(data, method=method)
    dendrogram(linked, orientation='top',
              distance_sort='descending',
              show_leaf_counts=True)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

#######################################
# 4. Cluster Interpretation & Insights
def interpret_clusters(pca_obj, pca_scores, clusters, original_df, n_features=5):
    """Generate cluster profiles and actionable insights"""
    # Create cluster profiles
    cluster_df = original_df.copy()
    cluster_df['Cluster'] = clusters
    # 1. Profile Table (Means/Medians)
    profile_table = cluster_df.groupby('Cluster').agg(['mean', 'median'])
    print("\nCluster Profiles:")
    display(profile_table)
    # 2. 2D Visualization with Centroids
    centroids = pd.DataFrame(pca_obj.transform(
        cluster_df.groupby('Cluster').mean()),
        columns=[f'PC{i+1}' for i in range(pca_scores.shape[1])])
    plt.figure(figsize=(10,8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster',
                   data=pca_scores.assign(Cluster=clusters),
                   palette='viridis', alpha=0.6)
    sns.scatterplot(x='PC1', y='PC2', data=centroids,
                   color='black', marker='X', s=200, label='Centroids')
    plt.title("Cluster Visualization with Centroids")
    plt.legend()
    plt.show()
    # 3. Key Drivers per Cluster
    print("\nKey Drivers per Cluster:")
    for cluster in np.unique(clusters):
        print(f"\nCluster {cluster}:")
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster].drop('Cluster', axis=1)
        overall_means = original_df.mean()
        # Find most distinctive features
        ratios = (cluster_data.mean() / overall_means).sort_values(ascending=False)
        print("Most elevated features:")
        print(ratios.head(n_features))
        print("\nMost depressed features:")
        print(ratios.tail(n_features))
    # 4. Cross-tabulation with InGroup (if exists)
    if 'InGroup' in original_df.columns:
        print("\nCluster vs InGroup Crosstab:")
        crosstab = pd.crosstab(cluster_df['Cluster'], original_df['InGroup'],
                              normalize='index')
        display(crosstab.style.background_gradient(cmap='Blues'))
        # Interpretation guidance
        print("\nInsights:")
        for cluster in crosstab.index:
            dominant_group = crosstab.loc[cluster].idxmax()
            print(f"Cluster {cluster} is {crosstab.loc[cluster, dominant_group]:.1%} aligned with {dominant_group}")
#######################################
# Main Execution Pipeline
# 1. Run PCA
pca, pca_scores, X_scaled = run_pca_analysis(X)

# 2. Determine optimal k
optimal_k = optimize_kmeans(pca_scores)
# 3. Perform K-means on PCA scores
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_scores)
# 4. Hierarchical clustering
hierarchical_clustering(pca_scores)
# 5. Generate insights
interpret_clusters(pca, pca_scores, clusters, X)
# 6. Interactive 3D visualization
if pca_scores.shape[1] >= 3:
    fig = px.scatter_3d(pd.DataFrame(pca_scores, columns=['PC1','PC2','PC3']).assign(Cluster=clusters),
                       x='PC1', y='PC2', z='PC3', color='Cluster',
                       title="3D Cluster Visualization")
    fig.show()
"""##Model Interpretability XAi##"""
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
# 1. Train XGBoost (replace with your actual data)
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)  # Use your pre-split data
# 2. Global Explanations (Feature Importance)
def global_explanations(model, X, y):
    """SHAP summary plot and permutation importance"""
    plt.figure(figsize=(15, 6))
    # SHAP Summary Plot
    plt.subplot(1, 2, 1)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Global Feature Impact", fontsize=12)
    # Permutation Importance
    plt.subplot(1, 2, 2)
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X.columns[sorted_idx])
    plt.title("Permutation Importance", fontsize=12)
    plt.tight_layout()
    plt.show()
    return shap_values
# 3. Local Explanations (Single Prediction)
def local_explanations(model, X, instance_idx):
    """LIME and SHAP force plot for one prediction"""
    # LIME
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values, feature_names=X.columns, mode='regression')
    lime_exp = lime_explainer.explain_instance(
        X.iloc[instance_idx], model.predict, num_features=5)
    # SHAP Force Plot
    shap_explainer = shap.Explainer(model)
    shap_values = shap_explainer(X.iloc[[instance_idx]])
    # Plot both
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    lime_exp.as_pyplot_figure()
    plt.title(f"LIME: Instance {instance_idx}")
    plt.subplot(1, 2, 2)
    shap.plots.force(
        shap_explainer.expected_value,
        shap_values.values[0],
        X.iloc[instance_idx],
        matplotlib=True)
    plt.title(f"SHAP: Instance {instance_idx}")
    plt.tight_layout()
    plt.show()
# Run XAI
print("=== Global Explanations ===")
shap_values = global_explanations(model, X_test, y_test)
print("\n=== Local Explanation (First Test Instance) ===")
local_explanations(model, X_test, 0)
# Advanced: SHAP Dependence Plot (for top feature)
print("\n=== SHAP Dependence Plot ===")
shap.plots.scatter(shap_values[:, 0], color=shap_values)
