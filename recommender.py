
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

def collaborative_recommender(df, user_index, num_recommendations=3):
    user_item = df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    n_users, n_items = user_item.shape
    k = min(20, n_users - 1, n_items - 1)
    if k < 1:
        return []

    U, sigma, Vt = svds(user_item.values, k=k)
    sigma = np.diag(sigma)
    pred_ratings = np.dot(np.dot(U, sigma), Vt)
    
    if user_index >= n_users:
        return []

    sorted_user_preds = np.argsort(pred_ratings[user_index])[::-1]
    known_items = user_item.iloc[user_index].to_numpy().nonzero()[0]
    recommended = [i for i in sorted_user_preds if i not in known_items]
    recommended_product_ids = user_item.columns[recommended][:num_recommendations]
    return recommended_product_ids

def content_based_recommender(df, product_id, top_n=3):
    if product_id not in df['product_id'].values:
        return pd.DataFrame()
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['product_description'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    index = df.index[df['product_id'] == product_id][0]
    sim_scores = list(enumerate(cos_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return df.iloc[sim_indices][['product_id', 'product_title']]
