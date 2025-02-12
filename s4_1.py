"""
Titanic Dataset Analysis
Author: Houssam Karroum
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Load Titanic dataset
data_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html"
titanic_df = pd.read_csv("titanic.csv")

# Display Code in Streamlit
def display_code(code):
    st.code(code, language="python")

# Streamlit App Title
st.title("Titanic Dataset Analysis & Author: Houssam Karroum")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Dataset Overview", "Data Cleaning", "Survival Analysis", "Correlation Analysis"])

# Dataset Overview
if section == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(titanic_df.head())
    st.write("### Missing Values")
    st.dataframe(titanic_df.isnull().sum())

# Data Cleaning
if section == "Data Cleaning":
    st.header("Data Cleaning")
    titanic_df = titanic_df.fillna(titanic_df.median(numeric_only=True))
    if "Sex" in titanic_df.columns:
        titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
        titanic_df = titanic_df.rename(columns={"Sex": "Genre"})
    st.write("### Cleaned Dataset Sample")
    st.dataframe(titanic_df.head())

# Survival Analysis
if section == "Survival Analysis":
    st.header("Survival Analysis")
    column_name = "Genre" if "Genre" in titanic_df.columns else "Sex"
    survival_by_gender = titanic_df.groupby(column_name)["Survived"].mean()
    fig, ax = plt.subplots()
    sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values, palette="coolwarm", ax=ax)
    ax.set_title("Survival Rate by Gender")
    ax.set_xlabel("Gender (0: Male, 1: Female)")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)

# Correlation Analysis
if section == "Correlation Analysis":
    st.header("Correlation Analysis")
    survivors = titanic_df[titanic_df['Survived'] == 1]
    non_survivors = titanic_df[titanic_df['Survived'] == 0]
    survivors_corr = survivors.select_dtypes(include=['number']).corr()
    non_survivors_corr = non_survivors.select_dtypes(include=['number']).corr()
    st.write("### Correlation Matrix - Survivors")
    fig, ax = plt.subplots()
    sns.heatmap(survivors_corr, annot=True, cmap='hot', ax=ax)
    st.pyplot(fig)
    st.write("### Correlation Matrix - Non-Survivors")
    fig, ax = plt.subplots()
    sns.heatmap(non_survivors_corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
