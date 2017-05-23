import pandas
from sklearn.cross_validation import train_test_split
import RecommendationEngine.SongRecommendations.Recommenders as rec

#Load the data
triplets_file = 'C:/Users/Madhu/Desktop/1000000.txt'
songs_metadata_file = 'C:/Users/Madhu/Desktop/song.csv'

song_df_1=pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df=pandas.merge(song_df_1,song_df_2.drop_duplicates(['song_id']),on="song_id",how="left")
song_df.head()

#print(len(song_df))
song_df = song_df.head(10000)


#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']


song_grouped=song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
grouped_sum=song_grouped['listen_count'].sum()
song_grouped['percentage']=song_grouped['listen_count'].div(grouped_sum)*100

song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
#print(song_grouped)


users = song_df['user_id'].unique()
songs = song_df['song'].unique()

#cross validtion data
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)



#Simple popularity-based recommender class (Can be used as a black box)
#Recommenders.popularity_recommender_py
pbr=rec.popularity_recommender_py()
pbr.create(train_data,'user_id','song')

#Use the popularity model to make some predictions¶

user_id = users[123]
output=pbr.recommend(user_id)
#print(output)


#Recommenders.item_similarity_recommender_py
#Create an instance of item similarity based recommender class

is_model = rec.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

user_id=users[5]
user_items=is_model.get_user_items(user_id)

print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
out=is_model.recommend(user_id)
#print(out)

#We can also apply the model to find similar songs to any song in the dataset
ot=is_model.get_similar_items(['U Smile - Justin Bieber'])
print(ot)