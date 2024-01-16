import streamlit as st
import os
import database
import streamlit_functions
import streamlit_vis

def main():
    # Set up the database
    # Check if the database already exists
    if not os.path.exists("data/movies.db"):
        print("Setting up database...")
        database.set_up_db(reviews_path="data/reviews.csv")

    # Set the title of the app
    st.title("My Movie Review App")
    st.markdown("This is a simple web app in which you can write reviews for your favorite movies.")
    
    # Create the windows for the app
    streamlit_functions.review_window()

    streamlit_functions.watchlist_window()

    streamlit_functions.review_table()

    streamlit_functions.get_watchlist()

    streamlit_functions.delete_window()

    streamlit_vis.visualisation_window()

if __name__ == "__main__":
    main()
