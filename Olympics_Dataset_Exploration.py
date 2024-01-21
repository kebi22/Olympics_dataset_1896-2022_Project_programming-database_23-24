#####################
# Import Libraries
#####################
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
#import seaborn as sb



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
# print(vendors_products_df.info())
# print(vendors_products_df.describe().T)
#####################
## Data Cleaning
#####################
# Apply the function to the Full_Name column
Athletes_df['Full_Name'] = Athletes_df['Full_Name'].apply(reformat_name)
Medals_df = Medals_df.dropna(subset=['Full_Name'])
Medals_df['Full_Name'] = Medals_df['Full_Name'].apply(reformat_name)

#Merge all the data frames thst sre significant to the Exploratory Data Analysis
Merged_df = pd.merge(Medals_df, Hosts_df[['game_slug','game_year', 'game_season']], 
                     left_on='Slug_Game', right_on='game_slug', 
                     how='left').drop(columns='game_slug')

#Final Merged Df
Olympics= pd.merge(Merged_df,Athletes_df[['Full_Name','TotalMedals_Won','Birth_year','Game_ParticipationS']],
                    left_on='Full_Name',right_on='Full_Name' ,how='left')
#Drop  NaN values
Olympics = Olympics.dropna(subset=['Full_Name'])




# Fill the NaN values of the athlete's birth year ith the median and change the tyoe to int
median_year_birth = Athletes_df['Birth_year'].median()
Olympics['Birth_year'].fillna(median_year_birth, inplace=True)
Olympics['Birth_year'] = Olympics['Birth_year'].astype(int)

# Olympics dataframe exploration
generic_write('So now  we have explored those 3 Data frames above and we will try to see how they can be merged to give a meaningful infromation')
write_subheader('Olympics')
generic_write('The whole Dataframe showcasing important features for the exploration and data analysis ')
write_df(Olympics)
