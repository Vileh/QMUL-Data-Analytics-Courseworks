import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import database

# The window containing all the plots
def visualisation_window():
    with st.expander("Visualise your reviews"):
        # Get all the reviews from the database
        imdb_vs_own()

        most_watched_genres()

        lineplot_movies_watched()

# Function plotting the IMDB rating vs. your score
def imdb_vs_own():
    reviews = database.fetch("SELECT reviews.movie_name, score, IMDB_rating FROM reviews INNER JOIN movies ON reviews.movie_id = movies.movie_id")
    df = pd.DataFrame(reviews, columns=["Movie Name", "Score", "IMDB Rating"])

    # Create an interactive scatter plot
    fig = px.scatter(df, x="IMDB Rating", y="Score", hover_data=['Movie Name'])
    fig.update_layout(title="IMDB Rating vs. Your Score")

    # Make the scatter dots bigger
    fig.update_traces(marker=dict(size=10))

    # Display the plot
    st.plotly_chart(fig)

# Function plotting the most watched genres
def most_watched_genres():
    genres = database.fetch("SELECT genre FROM movies INNER JOIN reviews ON movies.movie_id = reviews.movie_id")
    genre_dict = {}
    for movie_iter in genres:
        for j in movie_iter[0].split(', '):
            if j in genre_dict:
                genre_dict[j] += 1
            else:
                genre_dict[j] = 1
    
    fig = px.bar(x=list(genre_dict.keys()), y=list(genre_dict.values()))
    fig.update_layout(title="Most Watched Genres", xaxis_title="Genre", yaxis_title="Number of Movies")
    st.plotly_chart(fig)

# Function plotting the number of movies watched each month
def lineplot_movies_watched():
    # Number of movies watched  each month
    movies_watched = database.fetch("SELECT date_added FROM reviews")
    df = pd.DataFrame(movies_watched, columns=["Date Added"])
    df['Date Added'] = pd.to_datetime(df['Date Added'], format='%d/%m/%Y')
    df['Date Added'] = df['Date Added'].dt.strftime('%Y-%m')
    df = df.groupby(['Date Added']).size().reset_index(name='counts')

    fig = px.line(df, x="Date Added", y="counts")
    fig.update_layout(title="Movies Watched Each Month", xaxis_title="Month", yaxis_title="Number of Movies")
    fig.update_xaxes(tickformat="%Y-%m", dtick="M1")

    # Add invisible scatter trace with extra points at the start and end
    start_date = (pd.to_datetime(df['Date Added'].min(), format='%Y-%m') - pd.DateOffset(weeks=1))
    end_date = (pd.to_datetime(df['Date Added'].max(), format='%Y-%m') + pd.DateOffset(weeks=1))
    fig.add_scatter(x=[start_date, end_date], y=[0, 0], mode='markers', marker=dict(size=0, opacity=0), hoverinfo='none', showlegend=False)

    st.plotly_chart(fig)

# Expand!