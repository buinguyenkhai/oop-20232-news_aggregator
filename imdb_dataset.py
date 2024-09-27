import pandas as pd
'''
Class to get data from IMDb database
Class attributes: self.basics, self.directors, self.cast, self.ratings to merge with MovieLens database using imdbId
'''
class IMDbDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.basics = self.initializeBasics()
        self.directors = self.initializeDirectors()
        self.cast = self.initializeCast()
        self.ratings = self.initializeRatings()

    def initializeDataframe(self, file_name, fields, dtype=None):
        mylist = []
        for chunk in pd.read_csv(self.folder_path+file_name, sep='\t', usecols=fields, dtype=dtype, chunksize=20000, low_memory=False):
            mylist.append(chunk)
        df = pd.concat(mylist, axis= 0)
        del mylist
        return df
    
    def initializeRatings(self):
        ratings_field = ['tconst', 'averageRating']
        ratings = pd.read_csv(self.folder_path+'title.ratings.tsv.gz', sep='\t', usecols=ratings_field, low_memory=False)
        ratings.columns = ['imdbId', 'rating']
        return ratings

    def initializeBasics(self):
        basics_field = ['tconst', 'titleType', 'startYear', 'genres']
        basics = self.initializeDataframe('title.basics.tsv.gz', fields=basics_field)
        basics.columns = ['imdbId', 'titleType', 'year', 'genres']
        basics = basics[basics['titleType'] == 'movie']
        del basics['titleType']
        return basics
    
    def initializeNames(self):
        names_field = ['nconst', 'primaryName']
        names = self.initializeDataframe('name.basics.tsv.gz', fields=names_field)
        names.columns = ['personId', 'name']
        return names

    def initializeCast(self):
        cast_field = ['tconst', 'nconst', 'category']
        cast_dtype = {'category':'category'}
        cast = self.initializeDataframe('title.principals.tsv.gz', fields=cast_field, dtype=cast_dtype)
        cast.columns = ['imdbId', 'personId', 'job']
        cast = cast[(cast['job'].astype('category') == 'actor') | (cast['job'].astype('category') == 'actress')]
        del cast['job']
        cast['personId'] = cast['personId'].str[2:]
        cast.drop_duplicates(subset=['personId'], inplace=True)
        cast.rename(columns={'personId':'actorId'}, inplace=True)
        return cast

    def initializeDirectors(self):
        directors_field = ['tconst', 'directors']
        directors = self.initializeDataframe('title.crew.tsv.gz', fields=directors_field)
        directors.columns = ['imdbId', 'personId']
        directors['personId']= directors['personId'].apply(lambda x: x.split(',')[0] if "," in x else x)
        directors['personId']= directors['personId'].str[2:]
        directors['personId'] = pd.to_numeric(directors['personId'], errors='coerce')
        directors['personId'] = directors['personId'].astype('Int64')
        directors.rename(columns={'personId':'directorId'}, inplace=True)
        return directors
