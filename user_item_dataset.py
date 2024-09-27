import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV, KFold

# dataset for collaborative filtering
class UserItemDataset:
    def __init__(self, rating_df, min_rating_threshold, min_movie_threshold):
        # initialize a new dataset from a Rating Dataframe
        mat = user_item_matrix(rating_df, min_rating_threshold, min_movie_threshold)
        
        self.users = mat.columns
        self.movies = mat.index
        
        scaled_mat = mat.values - np.nanmean(mat.values, axis=0)
        scaled_mat[np.isnan(scaled_mat)] = 0
        self.mat = csr_matrix(scaled_mat)
        
    def get_user_watched_movies(self, userId):
        # get instances and ratings corresponding to movies watched by a specific user
        user_id_loc = self.users.get_loc(userId)
        watched_movie_id_loc = self.mat[:, user_id_loc] != 0
        not_id_loc = self.users != userId
        
        watched_movies = self.mat[watched_movie_id_loc.toarray().ravel()]
        
        X = watched_movies[:, not_id_loc]
        y = watched_movies[:, user_id_loc].toarray().ravel()
        
        return (X, y)
    
    def get_nominated_movies(self, userId):
        # get instances and ratings corresponding to nominated (unwatched) movies for a specific user
        user_id_loc = self.users.get_loc(userId)
        not_watched_movie_id_loc = self.mat[:, user_id_loc] == 0
        not_id_loc = self.users != userId
        
        not_watched_movies = self.mat[not_watched_movie_id_loc.toarray().ravel()]
        
        X = not_watched_movies[:, not_id_loc]
        
        return (X, self.movies[not_watched_movie_id_loc.toarray().ravel()])
    
    def customized_grid_search(self, users, estimator, param_grid, param_grid_size):
        # perform grid search for collaborative filtering
        final_rmse = np.zeros(param_grid_size)
        final_mae = np.zeros(param_grid_size)
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                                   scoring=('neg_root_mean_squared_error', 'neg_mean_absolute_error'), refit=False,
                                   cv=KFold(3, shuffle=True), n_jobs=-1)
        i = 1
        
        for user in users:
            try:
                X, y = self.get_user_watched_movies(user)
                grid_search.fit(X, y)
                rmse = grid_search.cv_results_['mean_test_neg_root_mean_squared_error']
                mae = grid_search.cv_results_['mean_test_neg_mean_absolute_error']
                final_rmse += rmse
                final_mae += mae
                print("Number of users evaluated:", i)
                i += 1
            except:
                pass
            
        param_df = pd.DataFrame(grid_search.cv_results_['params'])
        param_df['rmse'] = final_rmse / len(users)
        param_df['mae'] = final_mae / len(users)
            
        return param_df

def user_item_matrix(rating_df, min_rating_threshold, min_movie_threshold):
    # create user-item matrix
    qualified_movie = rating_df['movieId'].value_counts(sort=False) >= min_rating_threshold
    qualified_movie.name = 'qualified_movie'
    rating_df = rating_df.merge(qualified_movie, on='movieId')
    rating_df = rating_df[rating_df['qualified_movie']]

    qualified_user = rating_df['userId'].value_counts(sort=False) >= min_movie_threshold
    qualified_user.name = 'qualified_user'
    rating_df = rating_df.merge(qualified_user, on='userId')
    rating_df = rating_df[rating_df['qualified_user']]
    
    rating_df.drop(columns=['timestamp', 'qualified_movie', 'qualified_user'], axis=1, inplace=True)
    rating_df['rating'] = rating_df['rating'].astype(dtype='f4')
    rating_df['movieId'] = rating_df['movieId'].astype(dtype='i4')
    rating_df['userId'] = rating_df['userId'].astype(dtype='i4')
    
    return rating_df.groupby(['movieId', 'userId'])['rating'].sum().unstack().astype('f2')

if __name__ == '__main__':
    pass