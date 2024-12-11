import streamlit as st
import pandas as pd
import numpy as np

#
# Predicting Ratings with Item-Based Similarity
#
def generate_recommendations(user_ratings: pd.Series, similarity_frame: pd.DataFrame, backup_ids: pd.Series, all_movies: pd.DataFrame, top_k: int = 10) -> list[str]:
    """
    Given a user's rating vector, a similarity matrix, and a backup list of item IDs, 
    predict missing ratings and return top_k recommendations.

    Params:
    user_ratings : (pd.Series) contains user-provided ratings (1-5) or NaN.
    similarity_frame : (pd.DataFrame) square similarity matrix (items as both index and columns).
    backup_ids : (pd.Series) fallback list of movie IDs to use if we don't get enough predictions.
    all_movies : (pd.DataFrame) DataFrame with movie info (MovieID, Title, image_url, etc.)
    top_k : (int) Number of recommendations to return (default is 10).

    Returns: 
    (str) List of recommended movie IDs.
    """
    # consider only valid ratings (1 through 5), set others to NaN
    clean_ratings = user_ratings.where((user_ratings >= 1) & (user_ratings <= 5))
    predictions = clean_ratings.copy()

    # compute predicted ratings for all movies user didn't rate
    missing_indices = predictions.index[predictions.isna()]
    for movie_id in missing_indices:
        weighted_sum = similarity_frame.loc[movie_id].mul(clean_ratings, axis=0).sum(skipna=True)
        total_sim = similarity_frame.loc[movie_id][clean_ratings.notna()].sum(skipna=True)

        if total_sim != 0:
            predictions[movie_id] = weighted_sum / total_sim

    # sort by predicted rating (descending) for items not rated by user
    pred_candidates = predictions[clean_ratings.isna()].dropna().sort_values(ascending=False)
    final_recs = pred_candidates.index[:top_k].tolist()

    # if fewer than top_k, use backup list filtered to ensure presence in all_movies
    if len(final_recs) < top_k:
        already_used = set(clean_ratings.dropna().index).union(set(final_recs))
        valid_backup = [m for m in backup_ids if m not in already_used and m in all_movies["MovieID"].values]
        shortfall = top_k - len(final_recs)
        final_recs += valid_backup[:shortfall]

    return final_recs

#
# Load External Data and Resources
#
similarity_data_url = "https://github.com/djordjije/djordjije.github.io/raw/refs/heads/main/proj4/S100.parquet"
movie_data_url = "https://github.com/djordjije/djordjije.github.io/raw/refs/heads/main/proj4/top100_movies.csv"

# load similarity and top movies data
sim_matrix = pd.read_parquet(similarity_data_url)
all_movies = pd.read_csv(movie_data_url)
all_movies['MovieID'] = all_movies['MovieID'].astype(str)

# Ensure sim_matrix and all_movies have consistent movie sets
common_ids = sim_matrix.index.intersection(all_movies["MovieID"])
sim_matrix = sim_matrix.loc[common_ids, common_ids]
all_movies = all_movies[all_movies["MovieID"].isin(common_ids)].reset_index(drop=True)

# for demonstration, select first 15 movies to rate
initial_selection = all_movies.head(15)

#
# Streamlit UI Setup
#
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Movie Recommender")
st.subheader("Step 1: Rate as many movies as possible")

# Split the selected movies into two groups
front_section = initial_selection.iloc[:10]
additional_section = initial_selection.iloc[10:]

user_feedback = {}

with st.expander("Rate movies (click to expand/collapse)", expanded=True):
    st.write("### Movies to Rate")
    top_layout = st.columns(5)
    for idx, (_, item) in enumerate(front_section.iterrows()):
        with top_layout[idx % 5]:
            st.image(item["image_url"], width=150)
            rating_choice = st.radio(
                label=item["Title"],
                options=[None, 1, 2, 3, 4, 5],
                format_func=lambda x: "Select a rating" if x is None else "★" * x,
                key=item["MovieID"],
                index=0
            )
            user_feedback[item["MovieID"]] = rating_choice if rating_choice else 0

    st.write("### Scroll to Rate More Movies")
    extra_cols = st.columns(5)
    for idx, (_, row_data) in enumerate(additional_section.iterrows()):
        with extra_cols[idx % 5]:
            st.image(row_data["image_url"], width=150)
            rating_choice = st.radio(
                label=row_data["Title"],
                options=[None, 1, 2, 3, 4, 5],
                format_func=lambda x: "Select a rating" if x is None else "★" * x,
                key=row_data["MovieID"],
                index=0
            )
            user_feedback[row_data["MovieID"]] = rating_choice if rating_choice else 0


st.subheader("Step 2: Discover movies you might like")

if st.button("Click here to get your recommendations"):
    user_profile = pd.Series(index=sim_matrix.index, data=np.nan)
    for m_id, rate in user_feedback.items():
        if rate > 0:
            user_profile[m_id] = rate

    proposed_list = generate_recommendations(user_profile, sim_matrix, all_movies["MovieID"], all_movies, top_k=10)
    st.write("Your movie recommendations:")

    # Show 5 recommendations per row
    top_row = st.columns(5)
    for i in range(5):
        movie_id = proposed_list[i]
        info = all_movies.loc[all_movies["MovieID"] == movie_id]
        with top_row[i]:
            st.image(info["image_url"].values[0], width=150, caption=f"Rank {i+1}")
            st.write(info["Title"].values[0])

    bottom_row = st.columns(5)
    for i in range(5, 10):
        movie_id = proposed_list[i]
        info = all_movies.loc[all_movies["MovieID"] == movie_id]
        with bottom_row[i-5]:
            st.image(info["image_url"].values[0], width=150, caption=f"Rank {i+1}")
            st.write(info["Title"].values[0])
