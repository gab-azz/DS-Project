#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the CSV file with a semicolon delimiter
df = pd.read_csv("Proper_Treatment_Cleaned.csv", delimiter=';')
df1 = pd.read_csv("Proper_Control_Cleaned.csv", delimiter=';')

# Print the first 5 rows to verify successful loading
print("File loaded successfully. Here's a preview of the data:")
print(df.head())
print(df1.head())
# Print column names for further verification


#Descriptive Statistics for the Treatment and Control Group

treatment_data = pd.read_csv('Proper_Treatment_Cleaned.csv')
control_data = pd.read_csv('Proper_Control_Cleaned.csv')

# Standardize column names
treatment_data.columns = treatment_data.columns.str.strip().str.replace('\n', ' ')
control_data.columns = control_data.columns.str.strip().str.replace('\n', ' ')

# Columns of interest
columns_of_interest = [
    "Operating revenue (Turnover) th USD 2015",
    "Number of employees 2015",
    "Total Sales",
    "Sales in US/North America",
    "Sales in Europe/EMEA",
    "(Sales in Europe/EMEA)/ (Total Sales)"
]

# Function to calculate descriptive statistics
def descriptive_statistics(df, columns):
    stats = {}
    for column in columns:
        if column in df.columns:  # Ensure the column exists
            if column == "(Sales in Europe/EMEA)/ (Total Sales)":
                stats[column] = {
                    "Median": round(df[column].median(), 4),
                    "Mean": round(df[column].mean(), 4),
                    "Standard Deviation": round(df[column].std(), 4),
                    "Maximum": round(df[column].max(), 4),
                    "Minimum": round(df[column].min(), 4)
                }
            else:
                stats[column] = {
                    "Median": int(round(df[column].median())),
                    "Mean": int(round(df[column].mean())),
                    "Standard Deviation": int(round(df[column].std())),
                    "Maximum": int(round(df[column].max())),
                    "Minimum": int(round(df[column].min()))
                }
        else:
            stats[column] = "Column not found"
    return stats

# Calculate statistics for Treatment and Control datasets
treatment_stats = descriptive_statistics(treatment_data, columns_of_interest)
control_stats = descriptive_statistics(control_data, columns_of_interest)

# Print results in a cleaner format
print("Treatment Group Statistics:")
for column, stats in treatment_stats.items():
    print(f"\n{column}:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value}")

print("\nControl Group Statistics:")
for column, stats in control_stats.items():
    print(f"\n{column}:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value}")



#TWFE DiD analysis

import numpy as np
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

# PART 1: DATA PREPARATION
print("Step 1: Data Preparation\n")
# Load the dataset
data = pd.read_csv('DS_sample-26.11.csv')

# Fix DATE parsing
def parse_date(date):
    """Parses a date string into a datetime object with multiple format options."""
    for fmt in ('%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d'):
        try:
            return pd.to_datetime(date, format=fmt)
        except ValueError:
            continue
    return pd.NaT  # Return NaT if unparseable

data['DATE'] = data['DATE'].apply(parse_date)

# Drop rows with invalid DATE or critical missing data
data = data.dropna(subset=['DATE', 'RET', 'EuropePresence'])

# Ensure 'RET' is numeric
data['RET'] = pd.to_numeric(data['RET'], errors='coerce')
data = data.dropna(subset=['RET'])

# Add Interaction term
data['Interaction'] = data['PostGDPR'] * data['EuropePresence']

# Set multi-index for panel data
data = data.set_index(['TICKER', 'DATE'])
print("Data preparation completed.\n")

# PART 1.1: DESCRIPTIVE STATISTICS
print("Step 1.1: Descriptive Statistics for Returns (RET)")

# Calculate descriptive statistics for the full dataset
full_desc_stats = data['RET'].describe()[['mean', 'std', 'min', '50%', 'max']].rename({
    'mean': 'Mean',
    'std': 'Standard Deviation',
    'min': 'Minimum',
    '50%': 'Median',
    'max': 'Maximum'
})

# Filter data into treatment and control groups
treatment_group = data[data['EuropePresence'] == 1]
control_group = data[data['EuropePresence'] == 0]

# Calculate descriptive statistics for treatment group
treatment_desc_stats = treatment_group['RET'].describe()[['mean', 'std', 'min', '50%', 'max']].rename({
    'mean': 'Mean',
    'std': 'Standard Deviation',
    'min': 'Minimum',
    '50%': 'Median',
    'max': 'Maximum'
})

# Calculate descriptive statistics for control group
control_desc_stats = control_group['RET'].describe()[['mean', 'std', 'min', '50%', 'max']].rename({
    'mean': 'Mean',
    'std': 'Standard Deviation',
    'min': 'Minimum',
    '50%': 'Median',
    'max': 'Maximum'
})

# Print the descriptive statistics
print("\nDescriptive Statistics for Full Dataset:")
print(full_desc_stats)

print("\nDescriptive Statistics for Treatment Group:")
print(treatment_desc_stats)

print("\nDescriptive Statistics for Control Group:")
print(control_desc_stats)


# PART 2: TWO-WAY FIXED EFFECTS (TWFE) ANALYSIS
print("\nStep 2: Two-Way Fixed Effects (TWFE) Analysis\n")
# Fit the Two-Way Fixed Effects (TWFE) model
dependent = data['RET']
independent = data[['Interaction']]
model = PanelOLS(dependent, independent, entity_effects=True, time_effects=True)
results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

# Display the original results
print("TWFE Model Results:")
print(results.summary)

# PART 3: PARALLEL TRENDS CHECK
print("\nStep 3: Parallel Trends Check\n")
data['TreatmentGroup'] = data['EuropePresence']
average_returns = data.groupby(['DATE', 'TreatmentGroup'])['RET'].mean().unstack()
treatment_date = pd.Timestamp('2016-04-27')

# Plot parallel trends
plt.figure(figsize=(10, 6))
plt.plot(average_returns.index, average_returns[1], label='Treatment Group', linewidth=2)
plt.plot(average_returns.index, average_returns[0], label='Control Group', linewidth=2)
plt.axvline(x=treatment_date, color='red', linestyle='--', label='GDPR Announcement')
plt.title('Parallel Trends Check: Treatment vs Control')
plt.xlabel('Date')
plt.ylabel('Average Returns')
plt.legend()
plt.grid(True)
plt.show()

# PART 4: MONTE CARLO SIMULATION
print("\nStep 4: Monte Carlo Simulation\n")
n_simulations = 1000
random_effects = []

for i in range(n_simulations):
    # Randomly assign treatment while keeping the same number of treated firms
    treated_firms = data.index.get_level_values('TICKER').unique().to_series().sample(
        frac=data['EuropePresence'].mean(), random_state=i
    )
    data['RandomTreatment'] = data.index.get_level_values('TICKER').isin(treated_firms).astype(int)
    data['RandomInteraction'] = data['PostGDPR'] * data['RandomTreatment']

    # Fit the model with randomized treatment
    random_independent = data[['RandomInteraction']]
    random_model = PanelOLS(dependent, random_independent, entity_effects=True, time_effects=True)
    random_results = random_model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

    # Store the estimated treatment effect
    random_effects.append(random_results.params['RandomInteraction'])

    # Print progress every 100 simulations
    if (i + 1) % 100 == 0:
        print(f"Simulation {i + 1} completed.")

# Plot the distribution of random treatment effects
plt.figure(figsize=(10, 6))
plt.hist(random_effects, bins=30, alpha=0.7, color='blue', label='Randomized Effects')
plt.axvline(results.params['Interaction'], color='red', linestyle='--', label='Observed Effect')
plt.title('Monte Carlo Simulation: Randomized Treatment Effects')
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the p-value
p_value = (np.sum(np.abs(random_effects) >= np.abs(results.params['Interaction'])) / n_simulations)
print(f"\nP-value from Monte Carlo Simulation: {p_value}")

# SUMMARY
print("\nAnalysis Summary:")
print("1. Data Preparation: Successfully cleaned and structured the dataset for analysis.")
print("2. TWFE Analysis: Estimated the treatment effect using a Two-Way Fixed Effects model.")
print(f"   Observed Treatment Effect: {results.params['Interaction']}")
print("3. Parallel Trends Check: Visualized pre-treatment trends for validation.")
print(f"4. Monte Carlo Simulation: P-value = {p_value} to test the significance of the observed treatment effect.")






