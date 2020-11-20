'''
    <body background="ornekler/background.jpg">
    <body background="http://www.fikrin.net/grafik/logo.jpg">
    
        <style>
    body {
      background-image: url('back2.jpg');
      background-repeat: no-repeat;
      background-attachment: fixed;
      background-size: 100% 100%;
    }
    
    </style>
'''

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from surprise import SVD
from surprise import accuracy
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from collections import defaultdict

df_filtered = pd.read_csv('df_filtered.csv', low_memory=False)
Titles = pd.read_csv('Titles.csv', low_memory=False)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_filtered[['User', 'Movie', 'Rating']], reader)
trainset, testset = train_test_split(data, test_size=.20)
print('Fit etmeden once')
algo=SVD(n_factors=100, n_epochs=30, biased=True, lr_all=0.005, reg_all=0.04)
algo.fit(trainset)
pred_testset = algo.test(testset)
print(accuracy.rmse(pred_testset))
all_data = data.build_full_trainset()
data_toPredict = all_data.build_anti_testset()
predictions = algo.test(data_toPredict)

def top_movies(predictions, n=10):

    recommended = defaultdict(list)

    for userid, movieid, _ , estimation, _ in predictions:
        recommended[userid].append((movieid, estimation))

    for userid, user_ratings in recommended.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        recommended[userid] = user_ratings[:n]

    return recommended
    
recommended = top_movies(predictions, n=10)

print(recommended['Furkan'])

def Recommend_Movies(User):
    print('**Recommended Movies for User ' + User + ':**')
    print()
    order=1
    l = []

    for movie_est in recommended[User]:
        #print(str(order)+')',Titles.loc[movie_est[0],'Name'], '(' + str(Titles.loc[movie_est[0],'Year']) + ')')
        #temp = str(order)+')',Titles.loc[movie_est[0],'Name'], '(' + str(Titles.loc[movie_est[0],'Year']) + ')'
        temp = str(order)+') '+Titles.loc[movie_est[0]-1,'Name']+' ('+str(Titles.loc[movie_est[0]-1,'Year'])+')'
        l.append(temp)
        order+=1
    
    return l


    
#Recommend_Movies('Furkan')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = Recommend_Movies(movie)
    #movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run()
    

'''
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# define a function that creates similarity matrix
# if it doesn't exist
def create_sim():
    data = pd.read_csv('data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim


# defining a function that recommends 10 most similar movies
def rcmd(m):
    m = m.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if m not in data['movie_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title']==m].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
'''
