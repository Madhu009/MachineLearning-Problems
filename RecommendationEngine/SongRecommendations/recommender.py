#Class for Popularity based Recommender System model
import numpy as np
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


#Class for Item similarity based(content based) Recommender System model

class item_similarity_recommender_py():

    def __init__(self):
        self.train_data=None
        self.user_id=None
        self.item_id=None
        self.cooccurence_matrix=None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None


    # Get unique items (songs) corresponding to a given user
    def get_user_items(self,user_id):

        user_data=self.train_data[self.train_data[self.user_id]==user_id]
        user_items=list(user_data[self.item_id].unique())

        return user_items


    # Get unique users for a given item (song)
    def get_item_users(self, item):

        item_data=self.train_data[self.train_data[self.item_id]==item]
        item_users=list(item_data[self.user_id].unique())

        return item_users

    # Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())

        return all_items

    # Create the item similarity based recommender system model

    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id


        # Construct cooccurence matrix

    def construct_cooccurence_matrix(self, user_songs, all_songs):
        ####################################
        # Get users for all songs in user_songs.
        ####################################
        user_songs_users = []
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        # Initialize the item cooccurence matrix of size
        # len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        #############################################################
        # Calculate similarity between user songs and all unique songs
        # in the training data
        #############################################################


        for i in range(0,len(all_songs)):
            # Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id]==all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())

            for j in range(0,len(user_songs)):
                # Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]

                # Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)

                # Calculate cooccurence_matrix[i,j] as Jaccard Index

                if len(users_intersection) != 0:
                    # Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)

                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0
        return cooccurence_matrix

    # Use the item similarity based recommender system model to
    # make recommendations
    def recommend(self, user):

        ########################################
        # A. Get all unique songs for this user
        ########################################
        user_songs = self.get_user_items(user)

        print("No. of unique songs for the user: %d" % len(user_songs))

        ######################################################
        # B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()

        print("no. of unique songs in the training set: %d" % len(all_songs))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #######################################################
        # D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

        return df_recommendations

    # Get similar items to given items
    def get_similar_items(self, item_list):

        user_songs = item_list

        ######################################################
        # B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()

        print("no. of unique songs in the training set: %d" % len(all_songs))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #######################################################
        # D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

        return df_recommendations