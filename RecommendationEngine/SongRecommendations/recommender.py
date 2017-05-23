#Class for Popularity based Recommender System model
class Popular_based_recommender():

    def __init__(self):
        self.popular_recommendations=None
        self.user_id=None
        self.item_id=None
        self.train_data=None

    # Create the popularity based recommender system model
    def build_model(self,train_data,user_id,item_id):

        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # Get a count of user_ids for each unique song as recommendation score
        train_data_song_count=train_data.groupby[self.item_id].agg({self.user_id: 'count'}).reset_index()
        train_data_song_count.rename(columns={'user_id': 'score'},inplace=True)

        # Sort the songs based upon recommendation score
        train_data_sort=train_data_song_count.sort_values(['score', self.item_id],ascending = [0,1])

        # Generate a recommendation rank based upon score
        train_data_sort['Rank']=train_data_sort["score"].rank(ascending=0, method='first')

        # Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)


        # Use the popularity based recommender system model to
        # make recommendations

    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id

        # Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations