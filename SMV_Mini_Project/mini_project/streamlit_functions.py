import streamlit as st
import pandas as pd
import sqlite3
import database

# The review creation window
def review_window():
    movies = get_movies()
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()

    with st.expander("Write a new review"):
        # Input fields for movie name, score, and review
        movie_name = st.selectbox("Movie Name", options=movies, key="review")
        if review_exists(movie_name):
            st.warning("You already wrote a review for this movie. If you want to update your review, please use the form below.")
        select_date = st.date_input("Select a date for the entry", key="review_date")
        select_date = select_date.strftime("%d/%m/%Y")
        score = st.slider("Score (1-10)", 1, 10)
        review = st.text_area("Review", placeholder="Write your review here")
        # Save the review to the database
        if movie_name and score:  # Check if all values are inputted
            if st.button("Submit"):
                if in_watchlist(movie_name):
                    cursor.execute("DELETE FROM watchlist WHERE movie_name = ?".format(movie_name,))
                    st.success("Movie crossed off from watchlist!")
                if review_exists(movie_name):
                    cursor.execute("UPDATE reviews SET date_added = ?, score = ?, review = ? WHERE movie_name = ?", (select_date, score, review, movie_name))
                    st.success("Review updated successfully!")
                else:
                    save_review(movie_name, score, review, select_date)
                    st.success("Review submitted successfully!")
        else:
            st.warning("Please fill in all the fields.")

    conn.commit()
    cursor.close()
    conn.close()

# The watchlist creation window
def watchlist_window():
    movies = get_movies()
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()

    with st.expander("Add a movie to your watchlist"):
        movie_name = st.selectbox("Movie Name", options=movies, key="watchlist")
        if review_exists(movie_name):
            st.warning("You already wrote a review for this movie. If you want to update your review, please use the form above.")
        if in_watchlist(movie_name):
            st.warning("This movie is already in your watchlist.")
        select_date = st.date_input("Select a date for the entry", key="watchlist_date")
        select_date = select_date.strftime("%d/%m/%Y")
        note = st.text_area("Note", placeholder="Write your note here")
        if movie_name and select_date:
            if st.button("Add to watchlist"):
                if review_exists(movie_name):
                    st.warning("You already wrote a review for this movie. If you want to update your review, please use the form above.")
                elif in_watchlist(movie_name):
                    cursor.execute("UPDATE watchlist SET date_added = ?, note = ? WHERE movie_name = ?", (select_date, note, movie_name))
                    st.success("Watchlist entry updated successfully!")
                else:
                    # Add to watchlist
                    cursor.execute('''INSERT INTO watchlist 
                                        (date_added, movie_name, note)
                                        VALUES (?, ?, ?)''', 
                                        (select_date, movie_name, note))
                    st.success("Movie added to watchlist successfully!")
        else:
            st.warning("Please fill in all the fields.")

    conn.commit()
    cursor.close()
    conn.close()

# The review/watch list deletion window
def delete_window():
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()

    with st.expander("Delete a review or a watch list entry"):
        # Create a dropdown select box with two options
        option = st.selectbox("Select an option", options=["Delete a review", "Delete a watch list entry"])

        if option == "Delete a review":
            # Select a movie from the review table
            movie_name = st.selectbox("Movie Name", options=get_movies("reviews"), key="delete_review")
            # Present the review
            review = database.fetch("SELECT review FROM reviews WHERE movie_name = '{}'".format(movie_name))
            if len(review) > 0:
                st.warning("Review: {}".format(review[0][0]))
            if movie_name:
                if st.button("Delete review"):
                    cursor.execute("DELETE FROM reviews WHERE movie_name = ?", (movie_name,))
                    st.success("Review deleted successfully!")
        elif option == "Delete a watch list entry":
            movie_name = st.selectbox("Movie Name", options=get_movies("watchlist"), key="delete_watchlist")
            # Present the note
            note = database.fetch("SELECT note FROM watchlist WHERE movie_name = '{}'".format(movie_name))
            if len(note) > 0:
                st.warning("Note: {}".format(note[0][0]))
            if movie_name:
                if st.button("Delete watchlist entry"):
                    cursor.execute("DELETE FROM watchlist WHERE movie_name = ?", (movie_name,))
                    st.success("Watchlist entry deleted successfully!")

    conn.commit()
    cursor.close()
    conn.close()

# Show the review table
def review_table():
    with st.expander("View reviews"):
        # Get all the reviews from the database
        reviews = database.fetch("SELECT date_added, movie_name, score, review FROM reviews")

        df = pd.DataFrame(reviews, columns=["Date Added", "Movie Name", "Score", "Review"])
        df.reset_index(drop=True, inplace=True)
        
        # Convert the DataFrame to HTML and remove the index
        df_html = df.to_html(index=False)

        # Display the DataFrame
        st.markdown(df_html, unsafe_allow_html=True)

# Show the watchlist
def get_watchlist():
    with st.expander("Check your watchlist"):
        # Get all the reviews from the database
        conn = sqlite3.connect("data/movies.db")
        cursor = conn.cursor()
        cursor.execute("SELECT date_added, movie_name, note FROM watchlist")
        watchlist = cursor.fetchall()
        cursor.close()
        conn.close()

        df = pd.DataFrame(watchlist, columns=["Date Added", "Movie Name", "Note"])
        df.reset_index(drop=True, inplace=True)
        
        # Convert the DataFrame to HTML and remove the index
        df_html = df.to_html(index=False)

        # Display the DataFrame
        if len(df) == 0:
            st.warning("Your watchlist is empty.")
        else:
            st.markdown(df_html, unsafe_allow_html=True)

# Auxiliary functions
def delete_watchlist_entry(movie_name):
    # Delete a watch list entry from the database
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM watchlist WHERE movie_name = ?", (movie_name,))
    conn.commit()
    cursor.close()
    conn.close()

def in_watchlist(movie_name):
    # Check if a movie is in the watchlist
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM watchlist WHERE movie_name = ?", (movie_name,))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count > 0


def review_exists(movie_name):
    # Check if a review for the movie already exists in the database
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM reviews WHERE movie_name = ?", (movie_name,))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count > 0

def save_review(movie_name, score, review, date):
    # Save the review to the database
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()
    cursor.execute("SELECT movie_id FROM movies WHERE movie_name = ?", (movie_name,))
    movie_id = cursor.fetchone()[0]
    cursor.execute('''INSERT INTO reviews 
                    (date_added, movie_id, movie_name, score, review)
                    VALUES (?, ?, ?, ?, ?)''',
                    (date, movie_id, movie_name, score, review))
    conn.commit()
    cursor.close()
    conn.close()

def get_movies(table="movies"):
    # Get movie names
    movies = database.fetch("SELECT movie_name FROM {}".format(table))
    movies = [movie[0] for movie in movies]

    return movies
