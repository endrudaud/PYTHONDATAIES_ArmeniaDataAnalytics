import pandas as pd
import pyreadstat
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML
import statsmodels.api as sm
import numpy as np

# --- GENERAL MODULE FUNCTIONS ---

def load_data(file_path):
    """
    Load the dataset with metadata from the specified file path.
    """
    df, meta = pyreadstat.read_sav(file_path, encoding='UTF-8', disable_datetime_conversion=True)
    return df, meta

def gather_variable_descriptions(meta):
    """
    Gather and format descriptions of variables from the metadata.
    """
    variable_descriptions = []
    for var_name in meta.column_names:
        if var_name in meta.variable_to_label:
            variable_label = meta.column_labels[meta.column_names.index(var_name)]
            value_labels = meta.value_labels.get(meta.variable_to_label.get(var_name), {})
            variable_descriptions.append({
                "Variable": var_name,
                "Description": variable_label,
                "Value Labels": ", ".join([f"{value}: {label}" for value, label in value_labels.items()])
            })
    return pd.DataFrame(variable_descriptions)

def display_paginated_table(df, page_size=10):
    """
    Display a DataFrame in a paginated format.
    """
    total_rows = df.shape[0]
    total_pages = total_rows // page_size + (1 if total_rows % page_size > 0 else 0)
    
    def display_page(page):
        start_row = page * page_size
        end_row = min(start_row + page_size, total_rows)
        display(HTML(df.iloc[start_row:end_row].to_html(index=False)))
    
    # Pagination widgets
    page_slider = widgets.IntSlider(value=0, min=0, max=total_pages-1, step=1, description='Page')
    page_slider.observe(lambda change: display_page(change['new']), names='value')
    
    display(page_slider)
    display_page(0)  # Display the first page

def extract_value_labels(meta, variables_to_explain):
    """
    Extract value labels for specified variables from the metadata.
    """
    value_label_list = []

    # Extract value: label mappings for each variable
    for variable in variables_to_explain:
        if variable in meta.variable_to_label:
            value_labels = meta.value_labels.get(meta.variable_to_label[variable], {})
            for value, label in value_labels.items():
                value_label_list.append({'Variable': variable, 'Value': value, 'Label': label})

    # Convert to DataFrame for easy viewing and export
    value_label_df = pd.DataFrame(value_label_list)
    return value_label_df

def save_value_labels(value_label_df, output_path='./data/value_label_mappings.csv'):
    """
    Save value labels to a CSV file.
    """
    value_label_df.to_csv(output_path, index=False)

def plot_armenia_data_distribution(df, meta, page_size=20):
    """
    Plot Armenia data distribution and display paginated summary table.
    """
    # Map COUNTRY values to their corresponding labels using the metadata
    country_labels = meta.value_labels.get(meta.variable_to_label.get('COUNTRY'), {})
    df['COUNTRY'] = df['COUNTRY'].map(country_labels)

    # Filter the dataset to include only Armenia data
    df_armenia = df[df['COUNTRY'] == 'Armenia'].copy()  # Use .copy() to avoid chained assignment

    # Convert RESPAGE into age segments
    df_armenia['RESPAGE_segmented'] = pd.cut(df_armenia['RESPAGE'], bins=[17, 35, 55, float('inf')], labels=['18-35', '36-55', '56+'])

    # List of variables to plot, including the newly segmented RESPAGE
    variables_to_plot = [
        'RESPAGE_segmented', 'FATEINLF','BUSINRUS','BUSINTUR', 'BUSINGA', 'NEIGHBOR', 
        'USSRDISS', 'EUSUPP', 'EEUSUPNA', 'EEUNSUW', 'EEUSUPW', 'MAINFRN', 
        'MAINENEM', 'PARTYSUPP', 'WORKTYP', 'PERSINC', 'EDUYRS', 'RFAEDUC', 
        'RMOEDUC', 'RELIMP', 'ECONSTN', 'MONYTOT'
    ]

    # Create a summary DataFrame to store the value counts for each variable
    summary_list = []

    # Set up the plot grid
    num_plots = len(variables_to_plot)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Loop through each variable and create a plot
    for i, variable in enumerate(variables_to_plot):
        # Copy the column to avoid modifying the original DataFrame
        df_armenia[variable + '_mapped'] = df_armenia[variable]
        
        # If the variable has value labels, selectively map them
        if variable in meta.variable_to_label:
            value_labels = meta.value_labels.get(meta.variable_to_label[variable], {})
            # Map only specific values that have a label, keep numeric values intact
            df_armenia[variable + '_mapped'] = df_armenia[variable].apply(lambda x: value_labels.get(x, x))
        
        # Store the summary of value counts in the summary list
        value_counts = df_armenia[variable + '_mapped'].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        value_counts['Variable'] = variable
        summary_list.append(value_counts)

        # Plot the distribution of the variable without ordering
        if not df_armenia[variable + '_mapped'].isnull().all():  # Check if the column is not empty
            sns.countplot(data=df_armenia, x=variable + '_mapped', ax=axes[i])
            axes[i].set_title(f'Distribution of {variable} in Armenia')
            axes[i].set_xlabel(variable)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
        else:
            axes[i].set_visible(False)  # Hide the plot if there's no data

    # Remove any empty subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Concatenate all the summaries into a single DataFrame
    summary_df = pd.concat(summary_list, ignore_index=True)

    # Paginate the display of the DataFrame
    def display_page(page):
        start_row = page * page_size
        end_row = min(start_row + page_size, len(summary_df))
        display(HTML(summary_df.iloc[start_row:end_row].to_html(index=False)))

    # Create a slider for pagination
    page_slider = widgets.IntSlider(value=0, min=0, max=len(summary_df)//page_size, step=1, description='Page')
    page_slider.observe(lambda change: display_page(change['new']), names='value')

    display(page_slider)
    display_page(0)  # Display the first page


# --- SECTION I MODULE FUNCTIONS ---

def preprocess_data_section1(df, relevant_vars):
    """
    Preprocess the data for Section I: Filter out invalid rows and map USSRDISS to binary.
    """
    df_filtered = df[(df[relevant_vars] >= 0).all(axis=1)].copy()
    df_filtered['USSRDISS_binary'] = df_filtered['USSRDISS'].map({1.0: 1, 2.0: 0})
    return df_filtered

def analyze_parties_vs_ussrdiss(df_filtered):
    """
    Analyze the relationship between political parties and views on USSR Dissolution.
    """
    parties = df_filtered.groupby('PARTYSUPP')['USSRDISS_binary'].mean()
    print("USSRDISS by Political Party (Mean of Binary):")
    print(parties)
    return parties

def analyze_income_vs_ussrdiss(df_filtered):
    """
    Calculate the correlation between Personal Income and View on USSR Dissolution.
    """
    income_corr = df_filtered['PERSINC'].corr(df_filtered['USSRDISS_binary'])
    print(f"\nCorrelation between Personal Income and View on USSR Dissolution: {income_corr:.4f}")
    return income_corr

def analyze_education_vs_ussrdiss(df_filtered):
    """
    Calculate the correlation between Education Years and View on USSR Dissolution.
    """
    education_corr = df_filtered['EDUYRS'].corr(df_filtered['USSRDISS_binary'])
    print(f"\nCorrelation between Education Years and View on USSR Dissolution: {education_corr:.4f}")
    return education_corr

def analyze_age_vs_ussrdiss(df_filtered):
    """
    Calculate the correlation between Age and View on USSR Dissolution.
    """
    age_corr = df_filtered['RESPAGE'].corr(df_filtered['USSRDISS_binary'])
    print(f"\nCorrelation between Age and View on USSR Dissolution: {age_corr:.4f}")
    return age_corr

def perform_logistic_regression_section1(df_filtered, variable):
    """
    Perform logistic regression between the binary view on USSR Dissolution and a given variable.
    """
    X = sm.add_constant(df_filtered[variable])
    model = sm.Logit(df_filtered['USSRDISS_binary'], X).fit()
    print(f"\nLogistic Regression: {variable} vs. View on USSR Dissolution")
    print(model.summary())
    return model

def plot_ussrdiss_distributions(df_filtered):
    """
    Plot the distribution of opinions on USSR Dissolution by age, income, and education.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(x=df_filtered['USSRDISS_binary'], y=df_filtered['RESPAGE'])
    plt.title('Age Distribution by View on USSR Dissolution')
    plt.xlabel('View on USSR Dissolution (0: Bad, 1: Good)')
    plt.ylabel('Age')

    plt.subplot(1, 3, 2)
    sns.boxplot(x=df_filtered['USSRDISS_binary'], y=df_filtered['PERSINC'])
    plt.title('Personal Income by View on USSR Dissolution')
    plt.xlabel('View on USSR Dissolution (0: Bad, 1: Good)')
    plt.ylabel('Personal Income')

    plt.subplot(1, 3, 3)
    sns.boxplot(x=df_filtered['USSRDISS_binary'], y=df_filtered['EDUYRS'])
    plt.title('Education Years by View on USSR Dissolution')
    plt.xlabel('View on USSR Dissolution (0: Bad, 1: Good)')
    plt.ylabel('Years in Education')

    plt.tight_layout()
    plt.show()


# --- SECTION II MODULE FUNCTIONS ---

def preprocess_data_section2(df):
    """
    Preprocess the data for Section II: Filter out invalid rows and map MAINENEM to specific countries or groups.
    """
    mainenem_mapping = {
        1.0: 'Abkhazia', 2.0: 'Afghanistan', 3.0: 'Arabia', 4.0: 'Argentina', 5.0: 'Armenia',
        6.0: 'Australia', 7.0: 'Austria', 8.0: 'Azerbaijan', 9.0: 'Baltic countries',
        10.0: 'Belgium', 11.0: 'Belarus', 12.0: 'Canada', 13.0: 'Cyprus', 14.0: 'Czech republic',
        15.0: 'Denmark', 16.0: 'Eastern European Countries', 17.0: 'Egypt', 18.0: 'Estonia',
        19.0: 'EU', 20.0: 'Finland', 21.0: 'France', 22.0: 'Great Britain', 23.0: 'Germany',
        24.0: 'Greece', 25.0: 'Iceland', 26.0: 'India', 27.0: 'Iran', 28.0: 'Iraq', 29.0: 'Ireland',
        30.0: 'Israel', 31.0: 'Italy', 32.0: 'Japan', 33.0: 'Kazakhstan', 34.0: 'Korea',
        35.0: 'Kyrgyzstan', 36.0: 'Latvia', 37.0: 'Lithuania', 38.0: 'Luxembourg', 39.0: 'Many',
        40.0: 'Mexico', 41.0: 'Moldova', 42.0: 'Muslim countries', 43.0: 'Neighbouring countries',
        44.0: 'Netherlands', 45.0: 'Norway', 46.0: 'Ossetia', 47.0: 'Palestine', 48.0: 'Poland',
        49.0: 'Portugal', 50.0: 'Romania', 51.0: 'Russia', 52.0: 'Slovakia', 53.0: 'Spain',
        54.0: 'Sweden', 55.0: 'Switzerland', 56.0: 'Syria', 57.0: 'Turkey', 58.0: 'UAE',
        59.0: 'Ukraine', 60.0: 'Uruguay', 61.0: 'USA', 62.0: 'Uzbekistan', 63.0: 'Venezuela',
        64.0: 'China', 65.0: 'Nagorno-Karabakh', 66.0: 'Sweden', 67.0: 'Georgia', 81.0: 'Everybody',
        82.0: 'Except Russia', 83.0: 'Foreigners', 84.0: 'Our country itself', 85.0: 'Globalism',
        86.0: 'Local government'
    }

    relevant_vars = ['MAINENEM', 'EUSUPP', 'MAINFRN', 'PARTYSUPP', 'EDUYRS']
    df_filtered = df[(df[relevant_vars] >= 0).all(axis=1)].copy()
    df_filtered['MAINENEM_mapped'] = df_filtered['MAINENEM'].map(mainenem_mapping)

    return df_filtered

def analyze_eu_vs_russia_enemy(df_filtered):
    """
    Analyze EU Integration Support vs. Perception of Russia as the Main Enemy.
    """
    eu_russia_enemy = pd.crosstab(df_filtered['EUSUPP'], df_filtered['MAINENEM_mapped'] == 'Russia', normalize='index')
    print("EU Integration Support vs. Perception of Russia as Armenia's Main Enemy:")
    print(eu_russia_enemy)
    return eu_russia_enemy

def analyze_russia_ally_vs_enemy(df_filtered):
    """
    Analyze Russia as Main Ally vs. Perception of Russia, Turkey, or Azerbaijan as Main Enemy.
    """
    russia_ally_enemy = pd.crosstab(df_filtered['MAINFRN'] == 'Russia', df_filtered['MAINENEM_mapped'], normalize='index')
    print("\nRussia as Main Ally vs. Perception of Russia, Turkey, or Azerbaijan as Main Enemy:")
    print(russia_ally_enemy)
    return russia_ally_enemy

def analyze_party_vs_turkey_enemy(df_filtered):
    """
    Analyze Nationalist/Conservative Parties vs. Perception of Turkey as Main Enemy.
    """
    turkey_enemy_party = pd.crosstab(df_filtered['PARTYSUPP'], df_filtered['MAINENEM_mapped'] == 'Turkey', normalize='index')
    print("\nNationalist/Conservative Parties vs. Perception of Turkey as Armenia's Main Enemy:")
    print(turkey_enemy_party)
    return turkey_enemy_party

def analyze_education_vs_russia_enemy(df_filtered):
    """
    Analyze correlation between Education Years and Perception of Russia as Armenia's Main Enemy.
    """
    education_russia_enemy_corr = df_filtered['EDUYRS'].corr(df_filtered['MAINENEM_mapped'] == 'Russia')
    print(f"\nCorrelation between Education Years and Perception of Russia as Armenia's Main Enemy: {education_russia_enemy_corr:.4f}")
    return education_russia_enemy_corr

def perform_logistic_regression_section2(df_filtered, independent_var):
    """
    Perform logistic regression between the perception of Russia as the main enemy and an independent variable.
    """
    X = sm.add_constant(df_filtered[independent_var])
    model = sm.Logit(df_filtered['MAINENEM_mapped'] == 'Russia', X).fit()
    print(f"\nLogistic Regression: {independent_var} vs. Perception of Russia as Main Enemy")
    print(model.summary())
    return model

def plot_relationships_section2(df_filtered):
    """
    Plot relationships between perception of Russia as the main enemy, EU support, and education years.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=df_filtered['MAINENEM_mapped'] == 'Russia', y=df_filtered['EUSUPP'])
    plt.title('EU Integration Support by Perception of Russia as Main Enemy')
    plt.xlabel('Perceive Russia as Main Enemy')
    plt.ylabel('EU Integration Support')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_filtered['MAINENEM_mapped'] == 'Russia', y=df_filtered['EDUYRS'])
    plt.title('Education Years by Perception of Russia as Main Enemy')
    plt.xlabel('Perceive Russia as Main Enemy')
    plt.ylabel('Years in Education')

    plt.tight_layout()
    plt.show()

# --- SECTION III MODULE FUNCTIONS ---

def preprocess_data_section3(df):
    """
    Preprocess the data for Section III: Filter out invalid rows and map MAINFRN and MAINENEM to specific countries or groups.
    """
    mainfrn_mapping = {
        1.0: 'Abkhazia', 2.0: 'Afghanistan', 3.0: 'Arabia', 4.0: 'Argentina', 5.0: 'Armenia',
        6.0: 'Australia', 7.0: 'Austria', 8.0: 'Azerbaijan', 9.0: 'Baltic countries',
        10.0: 'Belgium', 11.0: 'Belarus', 12.0: 'Canada', 13.0: 'Cyprus', 14.0: 'Czech republic',
        15.0: 'Denmark', 16.0: 'Eastern European Countries', 17.0: 'Egypt', 18.0: 'Estonia',
        19.0: 'EU', 20.0: 'Finland', 21.0: 'France', 22.0: 'Great Britain', 23.0: 'Germany',
        24.0: 'Greece', 25.0: 'Iceland', 26.0: 'India', 27.0: 'Iran', 28.0: 'Iraq', 29.0: 'Ireland',
        30.0: 'Israel', 31.0: 'Italy', 32.0: 'Japan', 33.0: 'Kazakhstan', 34.0: 'Korea',
        35.0: 'Kyrgyzstan', 36.0: 'Latvia', 37.0: 'Lithuania', 38.0: 'Luxembourg', 39.0: 'Many',
        40.0: 'Mexico', 41.0: 'Moldova', 42.0: 'Muslim countries', 43.0: 'Neighbouring countries',
        44.0: 'Netherlands', 45.0: 'Norway', 46.0: 'Ossetia', 47.0: 'Palestine', 48.0: 'Poland',
        49.0: 'Portugal', 50.0: 'Romania', 51.0: 'Russia', 52.0: 'Slovakia', 53.0: 'Spain',
        54.0: 'Sweden', 55.0: 'Switzerland', 56.0: 'Syria', 57.0: 'Turkey', 58.0: 'UAE',
        59.0: 'Ukraine', 60.0: 'Uruguay', 61.0: 'USA', 62.0: 'Uzbekistan', 63.0: 'Venezuela',
        64.0: 'China', 65.0: 'Nagorno-Karabakh', 66.0: 'Sweden', 67.0: 'Georgia', 81.0: 'Everybody',
        82.0: 'Except Russia', 83.0: 'Foreigners', 84.0: 'Our country itself', 85.0: 'Globalism',
        86.0: 'Local government'
    }

    mainenem_mapping = {
        8.0: 'Azerbaijan', 51.0: 'Russia', 57.0: 'Turkey'
    }

    relevant_vars = ['MAINFRN', 'RESPAGE', 'PERSINC', 'MAINENEM']
    df_filtered = df[(df[relevant_vars] >= 0).all(axis=1)].copy()
    df_filtered['MAINFRN_mapped'] = df_filtered['MAINFRN'].map(mainfrn_mapping)
    df_filtered['MAINENEM_mapped'] = df_filtered['MAINENEM'].map(mainenem_mapping)

    return df_filtered

def analyze_age_vs_russia_friend(df_filtered):
    """
    Analyze correlation between Age and Perception of Russia as Armenia's Main Friend.
    """
    age_russia_friend_corr = df_filtered['RESPAGE'].corr(df_filtered['MAINFRN_mapped'] == 'Russia')
    print(f"Correlation between Age and Perception of Russia as Armenia's Main Friend: {age_russia_friend_corr:.4f}")
    return age_russia_friend_corr

def analyze_income_vs_eu_usa_friend(df_filtered):
    """
    Analyze Higher-Income Families vs. Perception of EU/USA as Armenia's Main Friend.
    """
    high_income_friend = pd.crosstab(df_filtered['PERSINC'], df_filtered['MAINFRN_mapped'].isin(['EU', 'USA']), normalize='index')
    print("\nIncome vs. Perception of EU/USA as Armenia's Main Friend:")
    print(high_income_friend)
    return high_income_friend

def analyze_enemy_vs_russia_friend(df_filtered):
    """
    Analyze Perception of Turkey/Azerbaijan as Enemy vs. Perception of Russia as Armenia's Main Friend.
    """
    enemy_friend_crosstab = pd.crosstab(df_filtered['MAINENEM_mapped'].isin(['Turkey', 'Azerbaijan']), df_filtered['MAINFRN_mapped'] == 'Russia', normalize='index')
    print("\nPerception of Turkey/Azerbaijan as Enemy vs. Perception of Russia as Armenia's Main Friend:")
    print(enemy_friend_crosstab)
    return enemy_friend_crosstab

def perform_logistic_regression_section3(df_filtered, independent_var, dependent_var):
    """
    Perform logistic regression between an independent variable and the perception of a specific country as Armenia's main friend.
    Ensure that the data is numeric and suitable for logistic regression.
    """
    # Convert dependent variable to binary numeric (0 or 1)
    dependent_var_numeric = dependent_var.astype(int)
    
    # Ensure the independent variable is numeric
    X = sm.add_constant(df_filtered[independent_var].astype(float))

    # Drop rows with NaN or infinite values
    mask = np.isfinite(dependent_var_numeric) & np.isfinite(X).all(axis=1)
    dependent_var_numeric = dependent_var_numeric[mask]
    X = X[mask]

    # Perform logistic regression
    model = sm.Logit(dependent_var_numeric, X).fit()
    print(f"\nLogistic Regression: {independent_var} vs. Perception of {dependent_var.name}")
    print(model.summary())
    return model

def plot_relationships_section3(df_filtered):
    """
    Plot relationships between perception of Russia as the main friend, age, and income.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=df_filtered['MAINFRN_mapped'] == 'Russia', y=df_filtered['RESPAGE'])
    plt.title('Age Distribution by Perception of Russia as Main Friend')
    plt.xlabel('Perceive Russia as Main Friend')
    plt.ylabel('Age')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_filtered['MAINFRN_mapped'].isin(['EU', 'USA']), y=df_filtered['PERSINC'])
    plt.title('Income Distribution by Perception of EU/USA as Main Friend')
    plt.xlabel('Perceive EU/USA as Main Friend')
    plt.ylabel('Income')

    plt.tight_layout()
    plt.show()

# --- Section IV: Analysis of Opinions about the EU ---

def load_data(file_path):
    """
    Load the dataset with metadata from the specified file path.
    """
    df, meta = pyreadstat.read_sav(file_path, encoding='UTF-8', disable_datetime_conversion=True)
    return df, meta

def preprocess_data_section4(df, relevant_vars):
    """
    Preprocess the data for Section IV analysis:
    - Filter out rows where any relevant variable is below 0 (indicating error or non-response).
    - Convert EUSUPP into a binary variable for logistic regression.
    """
    df_filtered = df[(df[relevant_vars] >= 0).all(axis=1)].copy()
    df_filtered['EUSUPP_binary'] = df_filtered['EUSUPP'].apply(lambda x: 1 if x >= 4 else 0)
    return df_filtered

def analyze_age_vs_eu_support(df_filtered):
    """
    Analyze the correlation between age and positive opinion of the EU.
    """
    age_eu_support_corr = df_filtered['RESPAGE'].corr(df_filtered['EUSUPP'])
    print(f"Correlation between Age and Positive Opinion of the EU: {age_eu_support_corr:.4f}")
    return age_eu_support_corr

def analyze_party_vs_eu_support(df_filtered, liberal_pro_european_parties):
    """
    Analyze the support for liberal/pro-European parties vs. positive opinion of the EU.
    """
    df_filtered['PARTYSUPP_mapped'] = df_filtered['PARTYSUPP'].map(liberal_pro_european_parties)
    liberal_party_eu_support = pd.crosstab(df_filtered['PARTYSUPP_mapped'], df_filtered['EUSUPP_binary'], normalize='index')
    print("\nSupport for Liberal/Pro-European Parties vs. Positive Opinion of the EU:")
    print(liberal_party_eu_support)
    return liberal_party_eu_support

def analyze_education_vs_eu_support(df_filtered):
    """
    Analyze the correlation between education years and positive opinion of the EU.
    """
    education_eu_support_corr = df_filtered['EDUYRS'].corr(df_filtered['EUSUPP'])
    print(f"\nCorrelation between Education Years and Positive Opinion of the EU: {education_eu_support_corr:.4f}")
    return education_eu_support_corr

def analyze_religion_vs_eu_support(df_filtered):
    """
    Analyze the correlation between religious importance and opinion of the EU.
    """
    religion_eu_support_corr = df_filtered['RELIMP'].corr(df_filtered['EUSUPP'])
    print(f"\nCorrelation between Religious Importance and Opinion of the EU: {religion_eu_support_corr:.4f}")
    return religion_eu_support_corr

def perform_logistic_regression_section4(df_filtered, independent_var, dependent_var='EUSUPP_binary'):
    """
    Perform logistic regression for a given independent variable against the binary opinion of the EU.
    """
    X = sm.add_constant(df_filtered[independent_var])
    model = sm.Logit(df_filtered[dependent_var], X).fit()
    print(f"\nLogistic Regression: {independent_var} vs. Positive Opinion of the EU")
    print(model.summary())
    return model

def plot_relationships_section4(df_filtered):
    """
    Plot the relationships between opinions on the EU and various factors such as age, education, and religious importance.
    """
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(x=df_filtered['EUSUPP_binary'], y=df_filtered['RESPAGE'])
    plt.title('Age Distribution by Opinion of the EU')
    plt.xlabel('Opinion of the EU (Binary)')
    plt.ylabel('Age')

    plt.subplot(1, 3, 2)
    sns.boxplot(x=df_filtered['EUSUPP_binary'], y=df_filtered['EDUYRS'])
    plt.title('Education Years by Opinion of the EU')
    plt.xlabel('Opinion of the EU (Binary)')
    plt.ylabel('Years in Education')

    plt.subplot(1, 3, 3)
    sns.boxplot(x=df_filtered['EUSUPP_binary'], y=df_filtered['RELIMP'])
    plt.title('Religious Importance by Opinion of the EU')
    plt.xlabel('Opinion of the EU (Binary)')
    plt.ylabel('Religious Importance')

    plt.tight_layout()
    plt.show()

# --- End of Section IV Module Functions ---