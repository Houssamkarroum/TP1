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
st.title("Titanic Dataset Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Dataset Overview", "Data Cleaning", "Survival Analysis", "Correlation Analysis", "Additional Analysis"])

# Dataset Overview
if section == "Dataset Overview":
    st.header("Dataset Overview")

    code_snippet = """
import pandas as pd
titanic_df = pd.read_csv("titanic.csv")
titanic_df.head()
    """
    display_code(code_snippet)

    st.write("### Sample Data")
    st.dataframe(titanic_df.head())

    st.write("### Dataset Info")
    buffer = io.StringIO()
    titanic_df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("### Missing Values")
    st.dataframe(titanic_df.isnull().sum())

    st.write("### Data Types")
    st.dataframe(pd.DataFrame(titanic_df.dtypes, columns=["Data Type"]))

# Data Cleaning
if section == "Data Cleaning":
    st.header("Data Cleaning")

    code_snippet = """
# Handle missing values
titanic_df = titanic_df.fillna(titanic_df.median(numeric_only=True))

# Drop 'Name' column
titanic_df = titanic_df.drop(['Name'], axis=1)

# Convert 'Sex' to numerical values
if "Sex" in titanic_df.columns:
    titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
    titanic_df = titanic_df.rename(columns={"Sex": "Genre"})

titanic_df.head()
    """
    display_code(code_snippet)

    # Handle missing values
    titanic_df = titanic_df.fillna(titanic_df.median(numeric_only=True))

    # Drop 'Name' column
    titanic_df = titanic_df.drop(['Name'], axis=1)

    # Convert 'Sex' to numerical values
    if "Sex" in titanic_df.columns:
        titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
        titanic_df = titanic_df.rename(columns={"Sex": "Genre"})

    st.write("### Cleaned Dataset Sample")
    st.dataframe(titanic_df.head())

# Survival Analysis
if section == "Survival Analysis":
    st.header("Survival Analysis")

    # Ensure 'Genre' column exists
    if "Genre" in titanic_df.columns:
        column_name = "Genre"
    else:
        column_name = "Sex"

    code_snippet = f"""
survival_by_gender = titanic_df.groupby("{column_name}")["Survived"].mean()

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values, palette="coolwarm", ax=ax)
ax.set_title("Survival Rate by Gender")
ax.set_xlabel("Gender (0: Male, 1: Female)")
ax.set_ylabel("Survival Probability")
plt.show()
    """
    display_code(code_snippet)

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

    code_snippet = """
# Separate data into survivors and non-survivors
survivors = titanic_df[titanic_df['Survived'] == 1]
non_survivors = titanic_df[titanic_df['Survived'] == 0]

# Select only numerical columns for correlation calculation
survivors_corr = survivors.select_dtypes(include=['number']).corr()
non_survivors_corr = non_survivors.select_dtypes(include=['number']).corr()

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.heatmap(survivors_corr, annot=True, cmap='hot', ax=ax)
plt.title("Correlation Matrix - Survivors")
plt.show()

fig, ax = plt.subplots()
sns.heatmap(non_survivors_corr, annot=True, cmap='coolwarm', ax=ax)
plt.title("Correlation Matrix - Non-Survivors")
plt.show()
    """
    display_code(code_snippet)

    # Separate data into survivors and non-survivors
    survivors = titanic_df[titanic_df['Survived'] == 1]
    non_survivors = titanic_df[titanic_df['Survived'] == 0]

    # Select only numerical columns for correlation calculation
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

# Additional Analysis
if section == "Additional Analysis":
    st.header("Additional Analysis")

    # Age Distribution
    st.subheader("Age Distribution")
    code_snippet = """
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Age Distribution
fig, ax = plt.subplots()
sns.histplot(titanic_df['Age'], kde=True, color='skyblue', ax=ax)
ax.set_title("Age Distribution of Passengers")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
plt.show()
    """
    display_code(code_snippet)

    fig, ax = plt.subplots()
    sns.histplot(titanic_df['Age'], kde=True, color='skyblue', ax=ax)
    ax.set_title("Age Distribution of Passengers")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Survival Rate by Class
    st.subheader("Survival Rate by Class")
    code_snippet = """
# Survival rate by class
survival_by_class = titanic_df.groupby("Pclass")["Survived"].mean()

import seaborn as sns
import matplotlib.pyplot as plt

# Plot survival rate by class
fig, ax = plt.subplots()
sns.barplot(x=survival_by_class.index, y=survival_by_class.values, palette="coolwarm", ax=ax)
ax.set_title("Survival Rate by Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Probability")
plt.show()
    """
    display_code(code_snippet)

    survival_by_class = titanic_df.groupby("Pclass")["Survived"].mean()

    fig, ax = plt.subplots()
    sns.barplot(x=survival_by_class.index, y=survival_by_class.values, palette="coolwarm", ax=ax)
    ax.set_title("Survival Rate by Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)

   

    # Survival Rate by Age Group
    st.subheader("Survival Rate by Age Group")
    code_snippet = """
# Create Age Groups
bins = [0, 12, 18, 35, 60, 100]
labels = ['Child', 'Teenager', 'Adult', 'Middle Aged', 'Elderly']
titanic_df['Age Group'] = pd.cut(titanic_df['Age'], bins=bins, labels=labels)

# Calculate survival rate by age group
survival_by_age_group = titanic_df.groupby("Age Group")["Survived"].mean()

import seaborn as sns
import matplotlib.pyplot as plt

# Plot survival rate by age group
fig, ax = plt.subplots()
sns.barplot(x=survival_by_age_group.index, y=survival_by_age_group.values, palette="coolwarm", ax=ax)
ax.set_title("Survival Rate by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("Survival Probability")
plt.show()
    """
    display_code(code_snippet)

    titanic_df['Age Group'] = pd.cut(titanic_df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teenager', 'Adult', 'Middle Aged', 'Elderly'])
    survival_by_age_group = titanic_df.groupby("Age Group")["Survived"].mean()

    fig, ax = plt.subplots()
    sns.barplot(x=survival_by_age_group.index, y=survival_by_age_group.values, palette="coolwarm", ax=ax)
    ax.set_title("Survival Rate by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)

    # Fare vs Survival
    st.subheader("Fare vs Survival")
    code_snippet = """
# Plot Fare vs Survival
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.boxplot(x="Survived", y="Fare", data=titanic_df, palette="coolwarm", ax=ax)
ax.set_title("Fare vs Survival")
ax.set_xlabel("Survived (0: No, 1: Yes)")
ax.set_ylabel("Fare")
plt.show()
    """
    display_code(code_snippet)

    fig, ax = plt.subplots()
    sns.boxplot(x="Survived", y="Fare", data=titanic_df, palette="coolwarm", ax=ax)
    ax.set_title("Fare vs Survival")
    ax.set_xlabel("Survived (0: No, 1: Yes)")
    ax.set_ylabel("Fare")
    st.pyplot(fig)

    # Gender and Class Interaction on Survival
    st.subheader("Gender and Class Interaction on Survival")
    code_snippet = """
# Survival rate by gender and class
survival_by_gender_class = titanic_df.groupby(["Pclass", "Sex"])["Survived"].mean()

import seaborn as sns
import matplotlib.pyplot as plt

# Plot survival rate by gender and class
fig, ax = plt.subplots()
sns.heatmap(survival_by_gender_class.unstack(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Survival Rate by Gender and Class")
plt.show()
    """
    display_code(code_snippet)

    survival_by_gender_class = titanic_df.groupby(["Pclass", "Sex"])["Survived"].mean()

    fig, ax = plt.subplots()
    sns.heatmap(survival_by_gender_class.unstack(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Survival Rate by Gender and Class")
    st.pyplot(fig)
