import pandas as pd
import sqlite3

# Function to preprocess the movies data
def preprocess_data():
    movies = pd.read_csv("data/imdb_top_1000.csv")
    movies = movies.drop(columns=['certificate', 'meta_score', 'gross'])
    movies['runtime'] = movies['runtime'].apply(lambda x: int(x.split(' ')[0]))

    movies = movies.rename(columns={"series_title": "movie_name"})

    # move the first column last
    cols = list(movies.columns)
    cols = cols[1:] + [cols[0]]
    movies = movies[cols]

    return movies


def set_up_db(reviews_path=None):
    # Create a database
    conn = sqlite3.connect("data/movies.db")
    # Create a cursor object
    cursor = conn.cursor()
    # Create the tables
    cursor.execute('''CREATE TABLE reviews 
                        (review_id INTEGER PRIMARY KEY AUTOINCREMENT, date_added DATE, movie_id INT, movie_name TEXT, 
                        score INT, review TEXT)''')
    cursor.execute('''CREATE TABLE watchlist
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, date_added DATE, movie_id INT, movie_name TEXT, note TEXT)''')
    
    # Set up the movies table
    movies_df = preprocess_data()
    # Add movies to the database
    movies_df.to_sql('movies', conn, if_exists='replace', index=True, index_label='movie_id')

    # Add the reviews if a path is given
    if reviews_path:
        reviews = pd.read_csv(reviews_path)       
        # Add reviews to the database
        for i in range(len(reviews)):
            name = reviews.iloc[i, 1]
            cursor.execute("SELECT movie_id FROM movies WHERE movie_name = '{}'".format(name))
            movie_id = cursor.fetchall()[0][0]
            cursor.execute('''INSERT INTO reviews 
                            (date_added, movie_id, movie_name, score, review)
                            VALUES (?, ?, ?, ?, ?)''',
                            (reviews.iloc[i, 0], movie_id, reviews.iloc[i, 1], int(reviews.iloc[i, 2]), reviews.iloc[i, 3]))
        
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()


def fetch(query):
    conn = sqlite3.connect("data/movies.db")
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result