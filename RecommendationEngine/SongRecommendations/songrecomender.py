import pandas
from sklearn.cross_validation import train_test_split
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
#print(len(song_df))

#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']


song_grouped=song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
grouped_sum=song_grouped['listen_count'].sum()
song_grouped['percentage']=song_grouped['listen_count'].div(grouped_sum)*100

song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
#print(song_grouped)


users = song_df['user_id'].unique()
print(len(users))

songs = song_df['song'].unique()
print(len(songs))


#cross validtion data
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
print(len(song_df))