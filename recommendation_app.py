import streamlit as st
df=read_csv("/Users/aishaqureshi/Desktop/Recommendation App/song_dataset.csv")
def main():
st.title("My Streamlit App")
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Create a user-song matrix
user_song_matrix = df.pivot_table(index='user', columns='song', values='play_count', fill_value=0)

# Normalize the play_count values
scaler = MinMaxScaler()
normalized_matrix = pd.DataFrame(scaler.fit_transform(user_song_matrix), columns=user_song_matrix.columns, index=user_song_matrix.index)

# Calculate the similarity between users
user_similarity = cosine_similarity(normalized_matrix)

def recommend_song(user_songs):
    # Find the user's index
    user_index = user_song_matrix.index.get_loc(user_songs.index[0])

    # Get the user's row from the similarity matrix
    user_row = user_similarity[user_index]

    # Find the user most similar to the given user
    similar_user_index = user_row.argmax()

    # Get the songs the similar user has listened to
    similar_user_songs = user_song_matrix.loc[user_song_matrix.index[similar_user_index]]

    # Find songs that the similar user has listened to but the given user has not
    recommended_songs = similar_user_songs[user_songs == 0]

    # Get the top recommended song
    top_song = recommended_songs.idxmax()

    return top_song

# Streamlit UI
st.title("Song Recommendation App")

# Dropdown for user to select songs
user_songs = st.multiselect("Select songs you have listened to:", user_song_matrix.columns)

if user_songs:
    recommended_song = recommend_song(pd.Series(user_songs))
    st.success(f"Recommended song: {recommended_song}")
else:
    st.info("Please select some songs to get recommendations.")

if __name__ == "__main__":
    main()
