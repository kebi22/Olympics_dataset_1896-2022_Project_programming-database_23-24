#####################
# Import Libraries
#####################
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns



# Function to visualize text to web page
def write_text(text):
    st.text(text)

# Function to visualize simple text to web page
def generic_write(text):
    st.write(text)

# Function to visualize dataframe to web page
def write_df(df):
    st.dataframe(df)

# Function to create a subheader to web page
def write_subheader(text):
    st.subheader(text)

# Function to create a title to web page
def write_title(text):
    st.title(text)

# Function to create a title to web page
def write_header(text):
    st.header(text)

# Function to write markdown code to web page
def write_md(text):
    st.markdown(text)

# Function to write caption to web page
def write_caption(text):
    st.caption(text)
# Function to reformat the Full_Name of the Athelete
def reformat_name(full_name):
    parts = full_name.split()
    # Check if there are at least two parts (first name and last name)
    if len(parts) > 1:
        first_name = parts[0].capitalize()
        # Capitalize each part of the last name which could be hyphenated or single
        last_name = '-'.join(sub.capitalize() for sub in parts[1].split('-'))
        # Combine the first name and the reformatted last name
        return f"{first_name} {last_name}"
    elif len(parts) == 1:
        # Handle cases with only one part in the name
        return parts[0].capitalize()
    else:
        # Handle empty strings or unexpected cases
        return full_name
#####################
# Main
#####################
## Reading datasets
Athletes_df = pd.read_csv('data/olympic_athletes.csv')
Hosts_df=pd.read_csv('data/olympic_hosts.csv')
Medals_df=pd.read_csv('data/olympic_medals.csv.csv')

# Initializing web page
write_title("Olympics_ dataset_Exploration")
generic_write('The project wants to analyze a dataset containing a list of Athelets and countries who participated in both the summer and winter game and won an olympic medal from 1896-2022')
generic_write('Every champion athelets with their respective countries are listed in the dataset')
generic_write('The main goal of the project is...')
generic_write('The  result can be very useful for .... ')
write_md('**Source dataset**:')
write_text('https://www.kaggle.com/datasets/piterfm/olympic-games-medals-19862018')
write_md('For more information read ```README.md``` file')

st.sidebar.write('Settings')
write_md('> Use the **sidebar menu** to show advanced features')

#####################
## Data Exploration
#####################
write_header('Data Exploration')

#####################

# Add expander to hide column information if not necessary
#'''with st.expander('Show column description'):
 #   for col in cve_df.columns:
  #      write_md(f"- ```{col.upper()}```: {cd.get_column_description('cve',col)}") # Using the external file to get column description


# print(cve_df.info())
# print(.describe().T)'''




#####################
# Athletes dataframe exploration
write_subheader('Atheletes DataFrame')
generic_write('Dataframe cointaining all Atheletes with their number of participation  and number of medals won')
write_df(Athletes_df.head())
# print(Athletes_df.info())
# print(Athletes_df.describe().T)

#####################
# Medals dataframe exploration
write_subheader('Medals DataFrame')
generic_write('Dataframe cointaining Medals with their respective champion athelets and gender ')
write_df(Medals_df.head())
# print(Medals_df.info())
# print(Medals_df.describe().T)

#####################
# Hosts dataframe exploration
write_subheader('Hosts DataFrame')
generic_write('Dataframe cointaining the countries that host the game and their respective game season')
write_df(Hosts_df.head())
# print(Hosts_df.info())
# print(Hosts_df.describe().T)
#####################
## Data Cleaning
#####################
# Apply the function to the Full_Name column
Athletes_df['Full_Name'] = Athletes_df['Full_Name'].apply(reformat_name)
Medals_df = Medals_df.dropna(subset=['Full_Name'])
Medals_df['Full_Name'] = Medals_df['Full_Name'].apply(reformat_name)

#Merge all the data frames thst are significant to the Exploratory Data Analysis
Merged_df = pd.merge(Medals_df, Hosts_df[['game_slug','game_year', 'game_season']], 
                     left_on='Slug_Game', right_on='game_slug', 
                     how='left').drop(columns='game_slug')

#Final Merged Df
Olympics= pd.merge(Merged_df,Athletes_df[['Full_Name','TotalMedals_Won','Birth_year','Game_ParticipationS']],
                    left_on='Full_Name',right_on='Full_Name' ,how='left')
#Drop  NaN values
Olympics = Olympics.dropna(subset=['Full_Name'])




# Fill the NaN values of the athlete's birth year with the median and change the type to int
median_year_birth = Athletes_df['Birth_year'].median()
Olympics['Birth_year'].fillna(median_year_birth, inplace=True)
Olympics['Birth_year'] = Olympics['Birth_year'].astype(int)
# Calculate the age of athletes by subtracting their birth year from the game year
Olympics['Age']=Olympics['game_year']-Olympics['Birth_year']

upper_age_limit = 65
lower_age_limit =12

# Replace negative ages and ages above the upper limit with NaN (or a chosen value/strategy)
Olympics.loc[Olympics['Age'] < lower_age_limit, 'Age'] = None
Olympics.loc[Olympics['Age'] > upper_age_limit, 'Age'] = None

mean_age = Olympics['Age'].mean()

Olympics['Age'].fillna(mean_age, inplace=True)


#Dropped the duplicated rows
Olympics=Olympics.drop_duplicates()

if st.sidebar.checkbox('Display final datasets'):
    write_header('Final datasets')
    generic_write('So now  we have explored those 3 Data frames above and we will try to see how they can be merged to give a meaningful infromation')
    generic_write('The whole Dataframe showcasing important features for the exploration and data analysis ')

    # Visualize Olympics Dataset
    write_subheader('Olympics')
    write_df(Olympics)

    with st.expander('Show activities done'):
        write_md("- Changed the Birth_year D_type from float to int")
        write_md("- Fill all NaN values of Birth_year with it's median_year_birth")
        write_md("- Merged the 3 data frames' columns which are essentil for the analysis")
        write_md("- Dropped the duplicated rows")

        write_subheader('Athletes_df and Medals_df')
        write_df(Athletes_df.head(5))
        write_df(Medals_df.head(5))
    with st.expander('Show activities done'):
        write_md("- Called the  function ```Reformat_name``` to the Full_Name column")
        write_md("- dropped all Nan values of the Full_name column ")


#####################
# Data Visualization
#####################

write_header('Plots')

# Plotting top 10 countries bty medals
write_subheader('Top 10 Countries by medals')
generic_write('What are top 10 countries to won the medals ')

top10=Olympics.Country.value_counts().head(10)
# Create the plot
fig, ax = plt.subplots(figsize=(22,10))
top10.plot(kind="bar", fontsize=15, ax=ax)
ax.set_title("Top 10 Countries by Medals", fontsize=15)
ax.set_ylabel("Medals", fontsize=14)


# Display the plot 
st.pyplot(fig) 

generic_write('Split the total medals of Top 10 Countries into Summer / Winter. Are there typical Summer/Winter Games Countries? ')
# Filter the merged dataframe to include only the top 10 countries by medals
olympics_10 = Olympics[Olympics.Country.isin(top10.index)]


sns.set(font_scale=1.5, palette="dark")
# Create the figure and the axes
fig, ax = plt.subplots(figsize=(20, 10))
# Create the countplot
ax = sns.countplot(data=olympics_10, x="Country", hue="game_season", order=top10.index)
# Set the title and labels
ax.set_title("Top 10 Countries by Medals", fontsize=20)
# Rotate the x-axis labels
plt.xticks(rotation=45)
#  display the figure
st.pyplot(fig)

generic_write('Split the total medals of Top 10 Countries into Gold, Silver, Bronze.')
# Set the aesthetic style of the plots
sns.set(font_scale=1.5, palette="dark")
# Create the figure and axes for the plot
fig, ax = plt.subplots(figsize=(20, 10))
# Create the countplot
sns.countplot(data=olympics_10, x="Country", hue="Medal", order=top10.index,
              hue_order=["GOLD", "SILVER", "BRONZE"], palette=["gold", "silver", "brown"], ax=ax)
# Set the title of the plot
plt.title("Top 10 Countries by Medals", fontsize=20)
# Rotate the x-axis labels if needed (optional)
plt.xticks(rotation=45)
# Use Streamlit to display the figure
st.pyplot(fig)



# Function to create a histogram using Seaborn
def plot_histogram(data, column, title, xlabel, ylabel, bins=30, color='blue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column].dropna(), bins=bins, kde=False, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt


#Display Histogram
write_subheader('Total Athlete Medals Distribution')
medals_plot = plot_histogram(Olympics, 'TotalMedals_Won', 'Distribution of Total Medals Won by Athletes', 'Total Medals', 'Number of Athletes')
st.pyplot(medals_plot)







write_subheader('Does ```TotalMedals_won``` and ```Game_ParticipationS``` columns have reslation with respect to gender ')
medals_and_Participations_by_gender = Olympics.groupby('Gender')[['TotalMedals_Won', 'Game_ParticipationS']].sum()

# Create two columns for the layout
col1, col2 = st.columns(2)

# Use the first column to display the first chart or data
with col1:
    write_header("Total Medals Won by Gender")
    # Assuming you have a plot function or you can use a bar chart directly
    fig, ax = plt.subplots()
    medals_and_Participations_by_gender['TotalMedals_Won'].plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Use the second column to display the second chart or data
with col2:
   write_header("Game Participations by Gender")
   # Assuming you have a plot function or you can use a bar chart directly
   fig, ax = plt.subplots()
   medals_and_Participations_by_gender['Game_ParticipationS'].plot(kind='bar', ax=ax)
   st.pyplot(fig)


discipline_counts = Olympics['Descipline'].value_counts()

# Get the top 10 disciplines
top_disciplines = discipline_counts.head(10).index

# Filter the original DataFrame to only include the top disciplines
top_disciplines_df = Olympics[Olympics['Descipline'].isin(top_disciplines)]

# Group by 'Discipline' and 'Gender' and count the occurrences
gender_distribution = top_disciplines_df.groupby(['Descipline', 'Gender']).size().unstack()

# Plot the gender distribution for the top disciplines
# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the data
gender_distribution.plot(kind='bar', stacked=True, ax=ax)

# Set the title and labels
ax.set_title('Gender Distribution in Top 10 Disciplines')
ax.set_xlabel('Discipline')
ax.set_ylabel('Count')
ax.legend(title='Gender')

# Rotate the x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Adjust the layout to fit the labels
plt.tight_layout()

# Show the plot in Streamlit
st.pyplot(fig)



# Calculate the age of athletes by subtracting their birth year from the game year
Olympics['Age']=Olympics['game_year']-Olympics['Birth_year']








#####################
# Machine Learning
#####################

write_header('Machine Learning application')
write_md('The main goal is ')
write_md('The parameters used to classify are: ``````')
write_md('The algorithm used to classify the data is ``````')
write_subheader(' dataset')


#Feature preprocessing




