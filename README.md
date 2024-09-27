# Movie Recommendation System

- All datasets used: https://drive.google.com/drive/folders/1FkgButJTlPucM65LyKZp1gZntK5MuXM4?usp=drive_link
Please put the all_datasets folder in the Project directory
db folder contains the database for GUI

#### Matrix factorization
Install scikit-surprise using conda (for matrix factorization only):
- Install mini conda
- In the conda command prompt, input the following command:
"conda create -n myenvironment <dependencies>"
- Select the new environment in your code editor to import the dependencies.

matrix_fact_param:
- Install dependencies (pandas, numpy, scikit-surprise, scikit-learn, tabulate)
- Run the code blocks in respective order, record the best parameter of the model.

matrix_factorization.ipynb:
- Install dependencies (nhu tren)
- Provide the directory of the 'ratings.csv' from the dataset
-  Run the code blocks in respective order, change the i and j value of the last code block to choose which user-movie rating value to predict. If the user has rated the movie, r_ui show true rating, if the user has not rated the movie, r_ui will show NaN

#### Collaborative filtering

user_item_dataset:
get dataset for collaborative filtering
cf_tuning:
demo code for tuning param
cf_recommendation:
main code for recommendation using collaborative filtering

#### GUI
get_movies_db:
create movies database for GUI

run 'streamlit run gui.py' on terminal and go to http://localhost:8501/

#### Content-based

imdb_dataset:
Class to get data from IMDb database
Class attributes: self.basics, self.directors, self.cast, self.ratings to merge with MovieLens database using imdbId

Codes to process data in order:
1. data_adding
2. data_cleaning
3. data_preprocessing

content-based:
main code to train and evaluate content-based models
