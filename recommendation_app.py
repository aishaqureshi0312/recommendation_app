import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the song dataset
df = pd.read_csv("/Users/aishaqureshi/Desktop/recommendation_app/song_dataset.csv")

def main():
    st.title("My Streamlit App")

    # Create a user-song matrix
    user_song_matrix = df.pivot_table(index='user', columns='song', values='play_count', fill_value=0)

    # Normalize the play_count values
    scaler = MinMaxScaler()
    normalized_matrix = pd.DataFrame(scaler.fit_transform(user_song_matrix), columns=user_song_matrix.columns, index=user_song_matrix.index)

    # Calculate the similarity between users
    user_similarity = cosine_similarity(normalized_matrix)

    st.title("Song Recommendation App")

    # Dropdown for user to select songs
    user_songs = st.multiselect("Select songs you have listened to:", user_song_matrix.columns)

    if user_songs:
        recommended_song = recommend_song(pd.Series(user_songs))
        st.success(recommended_song)
    else:
        st.info("Please select some songs to get recommendations.")

def recommend_song(user_songs):
    if not user_songs:
        return "No songs selected"

    # Get the first song in the list
    first_song = user_songs[0]

    try:
        print(f"Selected song: {first_song}")
        
        # Get the user's index from the user_song_matrix
        user_index = user_song_matrix.index.get_loc(first_song)
        print(f"User index: {user_index}")

        # Get the user's row from the similarity matrix
        user_row = user_similarity[user_index]

        # Find the user most similar to the given user
        similar_user_index = user_row.argmax()

        # Get the songs the similar user has listened to
        similar_user_songs = user_song_matrix.loc[user_song_matrix.index[similar_user_index]]

        # Find songs that the similar user has listened to but the given user has not
        recommended_songs = similar_user_songs[user_songs == 0]

        if recommended_songs.empty:
            return "No recommendations available for the selected songs"

        # Get the top recommended song
        top_song = recommended_songs.idxmax()

        return f"Recommended song: {top_song}"

    except KeyError:
        print(f"Song '{first_song}' not found in the dataset")
        return f"Song '{first_song}' not found in the dataset"

if __name__ == "__main__":
    main()
