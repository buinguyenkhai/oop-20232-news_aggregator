import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from cf_recommendation_gui import get_cf_recommendation
import user_item_dataset
import lightgbm as lgb
from joblib import load

# save important variables/settings in session state so it won't get re-run
if 'user_ratings' not in st.session_state:
    st.session_state['user_ratings'] = {}
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False
if 'sorted_by' not in st.session_state:
    st.session_state['sorted_by'] = 'Default'
if 'only_rated' not in st.session_state:
    st.session_state['only_rated'] = False
if 'not_rated' not in st.session_state:
    st.session_state['not_rated'] = False
# datasets
if 'data_reduced' not in st.session_state:
    st.session_state['data_reduced'] = pd.read_csv("all_datasets/content_based/data_reduced.csv", index_col=0)
if 'data_scaled' not in st.session_state:
    st.session_state['data_scaled'] = pd.read_csv("all_datasets/content_based/data_scaled.csv", index_col=0)
if 'user_item_dataset' not in st.session_state:
    st.session_state['user_item_dataset'] = load("all_datasets/collaborative/user_item_dataset_50_50.sav")
# models
if 'collab_estimators' not in st.session_state:
    st.session_state['collab_estimators'] = {
    'Collab-kNN':
    make_pipeline(TruncatedSVD(5), KNeighborsRegressor(20, weights='distance', metric='cosine')),
    'Collab-SVR':make_pipeline(TruncatedSVD(5), SVR(C=1.2, kernel='rbf')),
    'Collab-Stacking':make_pipeline(TruncatedSVD(50), StackingRegressor(estimators=[('knn', KNeighborsRegressor(20, weights='distance', metric='cosine')), ('svr', SVR(C=1.4))], final_estimator=Ridge(alpha=3.6), n_jobs=-1))
    }
if 'content_estimators' not in st.session_state:
    st.session_state['content_estimators'] = {
    'Content-Ridge':Ridge(alpha=90),
    'Content-kNN': KNeighborsRegressor(n_neighbors=16),
    'Content-RF': RandomForestRegressor(max_depth=10)
    }

conn = sqlite3.connect("db/movie_database.db")

# get X and y for user
def get_items_rated_by_user(data):
    user_ratings_df = pd.DataFrame(st.session_state['user_ratings'].items(), columns=['movieId', 'rating'])
    feature_vector = pd.merge(user_ratings_df, data, how='left', on='movieId')  
    return feature_vector.iloc[:, 2:], feature_vector['rating']

# use sql queries to get movies info to display on GUI
def fetch_movies(title, year_from, year_to, director, genre):
    if st.session_state['predicted']:
        query = """
    SELECT m.movieId, m.title, m.year, m.imdbId, m.rating, 
	    d.name, GROUP_CONCAT(g.genre, ' | ') AS genres, pr.predicted_rating
    FROM movie m
    LEFT JOIN director d ON m.directorId = d.directorId 
    LEFT JOIN genre g ON m.movieId = g.movieId
    LEFT JOIN predicted_ratings pr on pr.movieId = m.movieId
    WHERE 1 = 1
    """
    else:
        query = """
        SELECT m.movieId, m.title, m.year, m.imdbId, m.rating, 
            d.name, GROUP_CONCAT(g.genre, ' | ') AS genres
        FROM movie m
        LEFT JOIN director d ON m.directorId = d.directorId 
        LEFT JOIN genre g ON m.movieId = g.movieId
        WHERE 1 = 1
        """
    params = []
    if title:
        query += " AND m.title LIKE ?"
        params.append(f"%{title}%")
    if year_from:
        query += " AND m.year >= ?"
        params.append(year_from)
    if year_to:
        query += " AND m.year <= ?"
        params.append(year_to)
    if director:
        query += " AND d.name LIKE ?"
        params.append(f"%{director}%")
    if genre:
        query += " AND g.genre LIKE ?"
        params.append(f"%{genre}%")
    if st.session_state['only_rated']:
        query += " AND m.movieId IN (%s)" % ', '.join(['?'] * len(st.session_state['user_ratings']))
        params.extend(st.session_state['user_ratings'].keys())
    if st.session_state['not_rated']:
        query += " AND m.movieId NOT IN (%s)" % ', '.join(['?'] * len(st.session_state['user_ratings']))
        params.extend(st.session_state['user_ratings'].keys())

    query += " GROUP BY m.movieId"
    if st.session_state['sorted_by'] == 'Ascending':
        query += " ORDER BY pr.predicted_rating ASC"
    elif st.session_state['sorted_by'] == 'Descending':
        query += " ORDER BY pr.predicted_rating DESC"
    return pd.read_sql_query(query, conn, params=params)


def markdown_style(string, font_size, font_weight):
    return f"""
                <div style='
                    font-family: sans-serif; 
                    font-size: {font_size}; 
                    font-weight: {font_weight}; 
                    white-space: nowrap; 
                    overflow: hidden; 
                    text-overflow: ellipsis;
                    max-width: 250px;
                '>{string}</div>
                """

# display movies in GUI
def display_movies(movies):
    cols = st.columns(5)
    for index, row in movies.iterrows():
        with cols[index % 5]:
            st.markdown(markdown_style(row['title'], '17px', 'bold'), unsafe_allow_html=True)
            st.markdown(markdown_style(row['name'], '14px', 'normal'), unsafe_allow_html=True)
            st.markdown(markdown_style(row['genres'], '12px', 'normal'), unsafe_allow_html=True)
            st.markdown(markdown_style(f"IMDb rating: {row['rating']}", '12px', 'normal'), unsafe_allow_html=True)
            if st.session_state['predicted']:
                st.markdown(markdown_style(f"Predicted rating: {row['predicted_rating']}", '12px', 'bold'), unsafe_allow_html=True)
            rating = st.selectbox('Rate', options=['Select a rating',0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], index=0, key=row['movieId'])
            if rating != 'Select a rating':
                st.session_state['user_ratings'][(row['movieId'])] = rating

# main to run the app
def main():
    st.set_page_config(page_title="Movie Recommender", layout='wide')
    st.title('Movie Recommender System')

    st.sidebar.header('Settings')

    st.sidebar.subheader('Search Filters')
            
    title_filter = st.sidebar.text_input('Title')

    genres = ['Select','Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genre_filter = st.sidebar.selectbox('Select Genre', genres, index = 0)
    if genre_filter == 'Select':
        genre_filter = ''

    director_filter = st.sidebar.text_input('Director')

    year_from, year_to = st.sidebar.slider('Movies between:', 1895, 2023, (1895, 2023))

    st.session_state['sorted_by'] = st.sidebar.selectbox('Sort by predicted results', ['Default', 'Descending', 'Ascending'])

    only_rated = st.sidebar.checkbox("Show only rated movies")
    st.session_state['only_rated'] = only_rated
        
    not_rated = st.sidebar.checkbox("Show unrated movies")
    st.session_state['not_rated'] = not_rated

    if st.sidebar.button('Clear all ratings'):
        for movieId in st.session_state['user_ratings'].keys():
            st.session_state[str(movieId)] = 'Select a rating'
        st.session_state['user_ratings'].clear()

    st.sidebar.subheader('Choose Model')
    model_options = ['Collab-kNN', 'Collab-SVR', 'Collab-Stacking', 'Content-Ridge', 'Content-kNN', 'Content-RF', 'Content-LGBM']
    selected_model = st.sidebar.selectbox('Select a model', model_options)

    if st.sidebar.button('Predict'):
        st.session_state['predicted'] = True
        if st.session_state['predicted']:
            st.sidebar.write(f'Predictions using {selected_model}')
            # Collaborative models
            if selected_model[:6] == 'Collab':
                pred_movie_id, pred_rating = get_cf_recommendation(st.session_state['user_item_dataset'], st.session_state['user_ratings'], st.session_state['collab_estimators'][selected_model], len(st.session_state['user_item_dataset'].movies))
                pred_rating_and_movie_id = pd.DataFrame({'movieId': pred_movie_id, 'predicted_rating': pred_rating})
            # Content based models
            else:
                if selected_model == 'Content-Ridge':
                    data = st.session_state['data_scaled']
                else:
                    data = st.session_state['data_reduced']
                X, y = get_items_rated_by_user(data)
                # special case for LGBM
                if selected_model == 'Content-LGBM':
                    data = lgb.Dataset(X, y)
                    params = {
                        'objective': 'regression',
                        'metric': 'l2',
                        'boosting_type': 'gbdt',
                        'learning_rate': 0.01,
                        'num_leaves': 31,
                        'max_depth': -1,
                        'min_data_in_leaf': 5,
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                    }
                    model = lgb.train(params, data)
                else:
                    model = st.session_state['content_estimators'][selected_model]
                    model.fit(X, y)
                # predict on all movies
                predicted_rating = model.predict(data.drop(columns=['movieId']))
                predicted_rating = np.round(predicted_rating, 2)
                predicted_rating = np.clip(predicted_rating, 0.5, 5)
                pred_rating_and_movie_id = pd.DataFrame({'movieId': data['movieId'], 'predicted_rating': predicted_rating})
            # save to sql to display in GUI
            pred_rating_and_movie_id.to_sql('predicted_ratings', conn, index=False, if_exists='replace')
            movies = fetch_movies(title_filter, year_from, year_to, director_filter, genre_filter)
            display_movies(movies)

    st.header('Movies')
    movies = fetch_movies(title_filter, year_from, year_to, director_filter, genre_filter)
    display_movies(movies)


if __name__ == '__main__':
    main()  