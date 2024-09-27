import numpy as np
from numpy.random import choice

def dummy_ratings(user_item_dataset, size, score_lst):
    dummy_movies = choice(user_item_dataset.movies, size, replace=False)
    ratings = choice(score_lst, size)
    return dict(zip(dummy_movies, ratings))

def get_cf_recommendation(user_item_dataset, ratings, estimator, top_n):
    movie_vector = np.full(user_item_dataset.mat.shape[0], np.nan)
    movie_ids_loc = user_item_dataset.movies.isin(ratings)
    
    watched_movies = user_item_dataset.mat[movie_ids_loc]
    not_watched_movies = user_item_dataset.mat[~movie_ids_loc]
    not_watched_movie_ids = user_item_dataset.movies[~movie_ids_loc]
    
    for movie in ratings:
        try:
            movie_vector[user_item_dataset.movies.get_loc(movie)] = ratings[movie]
        except:
            pass
        
    rating_vector = movie_vector[~np.isnan(movie_vector)]
    average_rating = rating_vector.mean()
    rating_vector -= average_rating
    
    estimator.fit(watched_movies, rating_vector)
    predicted_ratings = estimator.predict(user_item_dataset.mat)
    best_movie_ids_loc = predicted_ratings.argsort()[:-top_n-1:-1]
    best_movie_ids = user_item_dataset.movies[best_movie_ids_loc]
    best_movie_scores = predicted_ratings[best_movie_ids_loc] + average_rating
    
    return best_movie_ids, best_movie_scores.round(4)

def display_recommendation(movie_df, best_movie_ids, best_movie_scores=None, set_index=True):
    if set_index:
        df = movie_df.set_index('movieId')
    best_movie_df = df.loc[best_movie_ids]
    if best_movie_scores is not None:
        best_movie_df.insert(len(best_movie_df.columns), 'predicted_rating', best_movie_scores)
    return best_movie_df
    
    
    