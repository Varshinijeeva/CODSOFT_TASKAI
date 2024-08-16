import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Sample data - replace this with your actual dataset
data = {
    'User': ['User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4', 'User5', 'User5'],
    'Paper': ['PaperA', 'PaperB', 'PaperA', 'PaperC', 'PaperB', 'PaperD', 'PaperC', 'PaperE', 'PaperD', 'PaperE'],
    'Interaction': [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]  # Binary interaction (e.g., view/like)
}

df = pd.DataFrame(data)

# Create a user-paper matrix
user_paper_matrix = df.pivot_table(index='User', columns='Paper', values='Interaction')

# Fill NaN values with 0 for similarity computation
user_paper_matrix_filled = user_paper_matrix.fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_paper_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_paper_matrix.index, columns=user_paper_matrix.index)

# Split the dataset into training and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

def create_user_paper_matrix(df):
    return df.pivot_table(index='User', columns='Paper', values='Interaction')

def compute_user_similarity(user_paper_matrix):
    user_paper_matrix_filled = user_paper_matrix.fillna(0)
    user_similarity = cosine_similarity(user_paper_matrix_filled)
    return pd.DataFrame(user_similarity, index=user_paper_matrix.index, columns=user_paper_matrix.index)

def recommend_papers(user, user_paper_matrix, user_similarity_df, top_n=3):
    # Get similar users
    similar_users = user_similarity_df[user].sort_values(ascending=False).index
    
    # Get paper interactions for similar users
    similar_users_interactions = user_paper_matrix.loc[similar_users]
    
    # Calculate weighted interactions
    weighted_interactions = similar_users_interactions.T.dot(user_similarity_df[user])
    similarity_sum = user_similarity_df[user].sum()
    
    # Avoid division by zero
    if similarity_sum == 0:
        similarity_sum = 1
    
    # Normalize by the sum of similarities
    weighted_interactions = weighted_interactions / similarity_sum
    
    # Exclude papers already interacted with by the user
    user_interacted_papers = user_paper_matrix.loc[user].dropna().index
    weighted_interactions = weighted_interactions.drop(user_interacted_papers, errors='ignore')
    
    # Get top N papers
    recommended_papers = weighted_interactions.sort_values(ascending=False)
    return recommended_papers.head(top_n)

def evaluate_model(user_paper_matrix, test_data, user_similarity_df):
    test_data_pivot = test_data.pivot_table(index='User', columns='Paper', values='Interaction')
    user_paper_matrix_filled = user_paper_matrix.fillna(0)
    
    y_true = []
    y_pred = []
    
    for _, row in test_data.iterrows():
        user = row['User']
        paper = row['Paper']
        true_interaction = row['Interaction']
        
        if user in user_paper_matrix.index and paper in user_paper_matrix.columns:
            predicted_interactions = recommend_papers(user, user_paper_matrix, user_similarity_df)
            
            if paper in predicted_interactions.index:
                predicted_interaction = predicted_interactions[paper]
            else:
                predicted_interaction = 0
            
            y_true.append(true_interaction)
            y_pred.append(predicted_interaction)
    
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Generate recommendations for a specific user
user = 'User1'
recommended_papers = recommend_papers(user, user_paper_matrix, user_similarity_df)
print(f"Recommended papers for {user}:")
print(recommended_papers)

# Evaluate the model
rmse = evaluate_model(user_paper_matrix, test_data, user_similarity_df)
print(f"Root Mean Squared Error (RMSE) on the test set: {rmse:.2f}")
