# %% [markdown]
# # <span style="color:rgb(94, 155, 221); font-size: 30px; font-weight: bold; padding: 10px; display: block; width: 100%;"> ☑ Netflix Case Study

# %% [markdown]
# <span style="color:rgb(94, 155, 221); font-size: 24px; font-weight: bold; border-left:3px solid; padding: 10px; display: block; width: 100%;"> Problem Statement
# 
# Analyze and predict the viewership and success of movies and TV shows on a streaming platform based on various attributes such as type, title, director, cast, country, date added, reinase year, rating, duration, listed in, and description
# 
# Specifically, I want to answer the following questions and perform tasks related to this dataset:
# 
# 
# 1. Viewership and Popularity Analysis:
# 
# - What are the most popular types of content on the platform (Movies or TV Shows)?
# 
# -  Which countres contribute the most content?
# 
# - Does the release year affect the popularity of content?
# 
# 2. Content Duration Analysis:
# 
# - Are there any trends in content duration over the years?
# 
# 3. Genre Analysis:
# 
# - Which genres are most prevalent on the platform?
# 
# - Are there any trends in the popularity of specific genres?
# 
# 4. Country Analysis:
# 
# - Do viewers have a preference for content from certain countries?
# 
# 5. Predictive Modeling:
# 
# - Can we build a model to pridict the success (eg, wewership, ratings) of a movie or TV show based on its attributes?
# 
# By addressing these questions and tasks, the streaming platform can gain insights into its content library, viewer preferences

# %% [markdown]
# <span style="color:rgb(94, 155, 221); font-size: 24px; font-weight: bold; border-left:3px solid; padding: 10px; display: block; width: 100%;"> Regarding Dataset
# 
# About this Dataset: Netflix is one of the most popular media and video streaming platforms. They have over 8000+ movies or tv shows available on their platform, as of mid-2021, thay have over 200M Subscribers globally. This tabular dataser consists of listings of all the movies and tv shows available on Netflix, along with details such as-cast, directors, ratings, release year, duration, etc.
# 
# Description of each column in the dataset:
# 
# 1. show id: A unique identifier for each show or movie.
# 
# 2. type: The type of content, either "Movie" or TV Show."
# 
# 3. title: The title of the movie or TV show.
# 
# 4. director: The director of the movie or TV show. In the first and third entries, this information is not available (NaN).
# 
# 5. cast: The cast or actors in the movie or TV show in the first entry, this information is not available (NaN). In the second entry, there is a list of actors from the TV show "Blood & Water"
# 
# 6. country: The country where the movie or TV show was produced or is assocsated with
# 
# 7. date_added: The date when the content was added to the streaming platform, in the forrnat "Month Day, Year"
# 
# 8. release year: The year the movie or TV show was origily released.
# 
# 9. rating: The content's rating, which indicates the recommended audiende age or maturity level (eg., "PD-13" or "TV-MA").
# 
# 10. duration: The duration of the movie or TV show in the first entry, the duration is given an minutes (190 min") in the second and third entries, it's indicated in the number of seasons ("2 Seasons" and "1 Season")
# 
# 11. listed in: The genre or category of the content, which can help classify It (e.g., "Documentaries," "International TV Shows," "Crime TV Shows"
# 
# 12. description: A brief description or synopsis of the movie or TV show, providing an overview of the pint or subject matter

# %% [markdown]
# <span style="color:rgb(94, 155, 221); font-size: 24px; font-weight: bold; border-left:3px solid; padding: 10px; display: block; width: 100%;"> Initiating Dataset Analysis
# 

# %%
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import plotly.express as px 
import plotly.graph_objects as go 
import seaborn as sns 

# %%
netflix_data = pd.read_csv(r"E:\0. VS Code\netflix_titles.csv\netflix_titles.csv")

# %%
netflix_data.head()

# %%
netflix_data.shape

# %%
netflix_data.info()

# %%
for col in netflix_data.columns:
    print(col)

# %%
netflix_data.isnull().sum()

# %%
netflix_data.count()

# %%
netflix_data.describe()

# %%
netflix_data["type"].unique()

# %%
for col in ["type", "title", "director", "country", "release_year", "rating",]:
    print(f"{col} has {netflix_data[col].unique()}\n\n")

# %% [markdown]
# <span style="color:rgb(94, 155, 221); font-size: 24px; font-weight: bold; border-left:3px solid; padding: 10px; display: block; width: 100%;"> Analyzing the Data

# %% [markdown]
# ***Netflix Data Null Percentage Analysis***
# 
# Perform an analysis of null (missing) values in a dataset, specificallya dataset related to Netflix content, the column names and their corresponding null percentage, making it easy to identfy which colomn has the most missing values.

# %%
null_percentage_df = pd.DataFrame(netflix_data.isnull().sum()/netflix_data.shape[0]*100, columns=["Null Percentage"]).reset_index().rename(columns={"index":"Column Names"}).sort_values(by="Null Percentage", ascending=False)
# Dividing missing values by total number of rows and multiplying by 100 to get percentage of missing values

null_percentage_df

# %%
null_count_df = pd.DataFrame( netflix_data.isnull().sum(), columns = ["Null Count"] ).reset_index().rename(columns={"index":"Column Names"}).sort_values(by="Null Count", ascending=False)
null_count_df

# %%
null_df = pd.merge(null_percentage_df, null_count_df, on="Column Names", how="inner")
null_df

# %% [markdown]
# ***Adviced to go by this way as you can easily correct any error in the code***

# %%
# Count of missing values in each column
null_count = netflix_data.isnull().sum()   

# Percentage of missing values in each column
null_percentage = netflix_data.isnull().sum()/netflix_data.shape[0]*100

# Creating a dataframe for missing values percentage
null_percentage_df = pd.DataFrame(null_percentage, columns=["Null Percentage"]).reset_index().rename(columns={"index":"Column Names"}).sort_values(by="Null Percentage", ascending=False)  

# Creating a dataframe for missing values count
null_count_df = pd.DataFrame(null_count, columns=["Null Count"]).reset_index().rename(columns={"index":"Column Names"}).sort_values(by="Null Count", ascending=False)   

# Merging both dataframes
null_df = pd.merge(null_percentage_df, null_count_df, on="Column Names", how="inner")   

null_df

# %% [markdown]
# The high percentage of missing values in the "director" and "cast" columns may impact certain analysis or recommendations that rely on this information. Depending on the specific goals of the analysis, it may be necessary to address these missing values through data imputation or to focus on aspects of the dataset that are more complete.

# %% [markdown]
# ***Imputation of Missing Values in Netflix Dataset***
# 
# Missing values in selected columns in Netflix data are imputed with the value "Unknown". The colomns chosen for imputation include: director, country, cast, rating and duration.

# %%
columns_to_impute = ["director", "country", "cast", "rating", "duration"]
for col in columns_to_impute:
    netflix_data[col].fillna("Unknown", inplace=True)

# %%
netflix_data.isnull().sum()

# %% [markdown]
# ***Segmentation of Netflix Data sets into TV Shows and Movies***
# 
# The Netflix dataset is segmented into two distinct subsets, TV shows and Movies. 
# 
# This segmentation is based on the "type" column, which indicates whether a given entry is "TV Show" or "Movie".

# %%
netflix_data.type.value_counts()

# %% [markdown]
# ***Finding the earliest and latest Movies and TV shows on Netflix***
# 

# %%
# Slicing data using the Movie title
movie_df = netflix_data[netflix_data["type"] == "Movie"]

oldest_movie = movie_df[movie_df['release_year'] == movie_df["release_year"].min()]

latest_movie = movie_df[movie_df['release_year'] == movie_df["release_year"].max()]

# %%
movie_df.head()

# %%
oldest_movie

# %%
latest_movie

# %%
# Slicing data using the TV Show title
tv_shows_df = netflix_data[netflix_data["type"] == "TV Show"]

oldest_tv_shows = tv_shows_df[tv_shows_df['release_year'] == tv_shows_df["release_year"].min()]

latest_tv_shows = tv_shows_df[tv_shows_df['release_year'] == tv_shows_df["release_year"].max()]

# %%
tv_shows_df.head()

# %%
oldest_tv_shows

# %%
latest_tv_shows

# %% [markdown]
# ***Finding top 5 popular genres on Netflix***
# 

# %%
# Slicing the listed in colomn to get the genres of the movies and TV shows
# Splitting the genres with (, ) 
# Exploding each records into different colomns
# And counting the number of times each genre appears
genre_count = netflix_data["listed_in"].str.split(', ').explode().value_counts() 
genre_count

# Plotting the top 5 popular genres on Netflix
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_count.head().index, y=genre_count.head().values, hue=genre_count.head().index, palette="viridis", legend=True)
# Using .head() to get the top 5 genres
# Using .index to get the genre names and .values to get the count of each genre 
plt.title("Top 5 Popular Genres on Netflix")
plt.xlabel("Genres")
plt.ylabel("Count")
plt.show()

# %%
# Plotting the popular genres on Netflix without using .head()
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_count.index, y=genre_count.values, hue=genre_count.index, palette="viridis")
plt.title("Popular Genres on Netflix")
plt.xlabel("Genres")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# Above we can see the top 5 most popular genre in the dataset, based on the number of titles available in netflix.

# %% [markdown]
# 

# %%
# Finding the lastest record in the dataset with respect to the release year using iloc() function
latest_record = netflix_data.iloc[netflix_data["release_year"].idxmax()]
latest_record

# %%
# Finding the oldest record in the dataset with respect to the release year using iloc() function
oldest_record = netflix_data.iloc[netflix_data["release_year"].idxmin()]
oldest_record

# %%
netflix_data.sort_values(by="release_year", ascending=False, inplace=True)


# %%
movie_df.sort_values(by="release_year", ascending=False, inplace=True)

# %%
recent_movies_data = movie_df[['title', 'type', 'release_year', 'country', 'description']].iloc[0]
pd.set_option('display.max_colwidth', None)
recent_movies_data

# %% [markdown]
# <span style="color:rgb(94, 155, 221); font-size: 24px; font-weight: bold; border-left:3px solid; padding: 10px; display: block; width: 100%;"> Visualizing the Data

# %% [markdown]
# ***Data Types Distribution of Netflix Dataset***
# 
# Specifically, this provides an insights into the number of coloumns in the dataset that belongs to each data type category (i.e., object, float, int, etc.)

# %%
%pip install nbformat

# %%
data_types_counts = netflix_data.dtypes.astype(str).value_counts()
fig = go.Figure(data=[go.Pie(labels=data_types_counts.index, values=data_types_counts.values, hole=0.5)])
fig.update_layout(title="Data Types Distribution in Netflix Dataset", width=500, height=500)
fig.show()

# %% [markdown]
# ***Netflix Content Release Year Distribution***
# 
# Generates a histogram using the Plotly Express Library to visualize the distribution of Netflix based on the release year. Each bar in the histrogram represents the count of content items released in the particular year.  

# %%
fig = px.histogram(netflix_data, x="release_year", title="Release Year Distribution in Netflix Dataset")
# Update the layout and add boxplot
fig.update_layout(bargap=0.1, bargroupgap=0.01, xaxis_title="Release Year", yaxis_title="Count")
fig.show()

# %% [markdown]
# ***Netflix Content by Country Distribution using Box Plot***
# 
# Creates a Histogram plot using Plotly Express Library (px) to visualize the distribution of Netflix Content by Country. I have used the "Country" coloumn on Y-Axis. Additionally, the plot includes a box plot marginals, which displays summary statistics (such as Quartiles and Outliers) for distribution of content within each country. 

# %%
ax = px.histogram(netflix_data, x="release_year", y="country", marginal="box", title="Countrywise Movie Release Distribution in Netflix Dataset")
ax.update_layout(bargap=0.01, bargroupgap=0.01, xaxis_title="Release Year", yaxis_title="Countries")
ax.show()

# %% [markdown]
# ***Analysis of Netflix Content Categories***
# 
# Performs an analysis of content categories within the Netflix dataset. It aims to provide insights into the distribution and popularity of different content categories.

# %%
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=netflix_data, y=netflix_data["listed_in"], hue=netflix_data["listed_in"], order=netflix_data["listed_in"].value_counts().index[0:25], palette="icefire", legend=False)
ax.bar_label(ax.containers[0])
plt.title("Top 20 Genres on Netflix")
plt.show()

# %% [markdown]
# # Miscellaneous
# 

# %%
# Finding out Top 5 Popular Countries on Netflix

# Splitting the country column
netflix_data["countries"] = netflix_data["country"].apply(lambda x: x.split(", "))
netflix_data["countries"].head()

# Creating a list of all countries
all_countries = []
for country in netflix_data["countries"]:
    all_countries.extend(country)

# Counting the frequency of each country
country_count = pd.Series(all_countries).value_counts()

# Plotting the top 5 countries
plt.figure(figsize=(12,6))
sns.barplot(x=country_count.head().index, y=country_count.head().values)
plt.xlabel("Countries")
plt.ylabel("Count")
plt.title("Top 5 Popular Countries on Netflix")
plt.show()

# %%
# Plotting the number of movies and TV shows released each year
plt.figure(figsize=(20, 10))
sns.countplot(x="release_year", hue="release_year", data=netflix_data, legend=False, palette="viridis")
plt.title("Number of Movies and TV Shows Released Each Year")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


