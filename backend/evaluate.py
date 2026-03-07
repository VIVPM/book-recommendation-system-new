import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add backend to path so we can import our configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from books_recommender.config.configuration import AppConfiguration

def load_ml_components():
    """
    Loads the trained KNN model and the data tables needed for recommendations.
    """
    print("Loading ML model and data...")
    config = AppConfiguration().get_recommendation_config()
    
    try:
        # Load the pivot table (rows = books, columns = users, values = ratings)
        book_pivot  = pickle.load(open(config.book_pivot_serialized_objects, "rb"))
        
        # Load the trained K-Nearest Neighbors model
        model = pickle.load(open(config.trained_model_path, "rb"))
        
        # Load the dataset containing user ratings
        final_rating = pickle.load(open(config.final_rating_serialized_objects, "rb"))
    except FileNotFoundError:
        print("\nError: Could not find the trained model or data files.")
        print("Please train the model first by hitting the `/train` endpoint.")
        raise FileNotFoundError("Could not find the trained model or data files.")
        
    print("Successfully loaded everything!\n")
    return book_pivot, model, final_rating

def test_recommendation_system(book_pivot, model, final_rating, number_of_recommendations=5):
    """
    Tests how good the recommendation system is using "Leave-One-Out" testing.
    
    How it works:
    1. Find a user who has given a high rating (8 out of 10 or higher) to MULTIPLE books.
    2. We take ONE of their favorite books and ask the model: "What 5 books are similar to this?"
    3. We check if the model's 5 recommendations include any of the OTHER books the user loved.
    4. If the model successfully recommends a book they actually read and loved, it's a HIT!
    """
    print(f"Testing the model by asking for {number_of_recommendations} recommendations per user...")
    
    # Step 1: Get only the books that users really liked (rating 7 or higher)
    loved_books_data = final_rating[final_rating['rating'] >= 7]
    
    # Create a dictionary where Key = User ID, Value = List of books they loved
    # Example: { User_123: ["Harry Potter", "Lord of the Rings"] }
    users_and_their_favorite_books = loved_books_data.groupby('user_id')['title'].apply(list).to_dict()
    
    # A list of all books our model actually knows about
    books_the_model_knows = set(book_pivot.index)
    
    # Variables to keep track of our scores
    total_users_tested = 0
    total_hits = 0
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    
    # To save time, we will only test 500 random users instead of thousands
    all_user_ids = list(users_and_their_favorite_books.keys())
    np.random.seed(42) # Keeps the random selection the same every time you run it
    np.random.shuffle(all_user_ids)
    users_to_test = all_user_ids[:500] 
    
    # Let's start testing!
    for user_id in tqdm(users_to_test, desc="Evaluating Users"):
        
        # Get all the books this specific user loved that our model knows about
        books_loved_by_user = [book for book in users_and_their_favorite_books[user_id] if book in books_the_model_knows]
        
        # We need the user to have loved at least 2 books to test them
        # (1 book to ask the model about, and at least 1 book to check if the model guessed right)
        if len(books_loved_by_user) < 2:
            continue
            
        # Let's take the first book they loved to give to the model (The Query)
        input_book = books_loved_by_user[0]
        
        # The rest of the books they loved are the "answers" we hope the model guesses (The Target)
        target_books = set(books_loved_by_user[1:])
        
        # --- ASK THE MODEL FOR RECOMMENDATIONS ---
        
        # Find exactly which row number our input book is in the pivot table
        row_number = np.where(book_pivot.index == input_book)[0][0]
        
        # Ask the KNN model for similar books
        distances, suggestions = model.kneighbors(
            book_pivot.iloc[row_number, :].values.reshape(1, -1), 
            n_neighbors=number_of_recommendations + 1
        )
        
        # The model always returns the input book itself as the #1 recommendation (because a book is 100% similar to itself)
        # So we skip index 0 and take the rest
        recommended_books = []
        for i in range(1, number_of_recommendations + 1):
            book_name = book_pivot.index[suggestions[0][i]]
            recommended_books.append(book_name)
            
        # --- GRADE THE MODEL'S ANSWERS ---
        
        # Did the model recommend any books that the user actually loved?
        correct_guesses = [book for book in recommended_books if book in target_books]
        
        if len(correct_guesses) > 0:
            total_hits += 1 # The model got at least one right!
            
        # Precision: Out of the 5 books we recommended, what percentage were correct?
        precision = len(correct_guesses) / number_of_recommendations
        precision_scores.append(precision)
        
        # Recall: Out of all the books the user loved, what percentage did we find?
        recall = len(correct_guesses) / len(target_books)
        recall_scores.append(recall)
        
        # NDCG: Measures ranking quality (did we put the correct books at the top?)
        # True relevance scores: 1 if the book was loved by the user, 0 if not
        true_relevance = [[1 if book in target_books else 0 for book in recommended_books]]
        
        # Predicted scores: We want the model to rank index 0 highest, index 4 lowest
        # We simulate this by giving decreasing scores based on position: [5, 4, 3, 2, 1]
        predicted_scores = [[number_of_recommendations - i for i in range(number_of_recommendations)]]
        
        from sklearn.metrics import ndcg_score
        
        # Only calculate NDCG if there is at least one correct guess, otherwise it's 0
        if len(correct_guesses) > 0:
            ndcg = ndcg_score(true_relevance, predicted_scores)
        else:
            ndcg = 0.0
            
        ndcg_scores.append(ndcg)
            
        total_users_tested += 1

    # --- PRINT THE FINAL REPORT CARD ---
    if total_users_tested == 0:
        print("Error: Could not find enough users with multiple high ratings to test.")
        return {"error": "Could not find enough users with multiple high ratings to test."}
        
    final_hit_ratio = total_hits / total_users_tested
    final_precision = np.mean(precision_scores)
    final_recall = np.mean(recall_scores)
    final_ndcg = np.mean(ndcg_scores)
    
    print(f"\n" + "="*40)
    print(f"         EVALUATION REPORT CARD         ")
    print(f"="*40)
    print(f"Users Tested: {total_users_tested}")
    print(f"Number of hits: {total_hits}")
    print(f"-"*40)
    print(f"Hit Ratio @ {number_of_recommendations}: {(final_hit_ratio*100):.2f}%")
    print(f"  ↳ Meaning: For {(final_hit_ratio*100):.2f}% of users, we successfully guessed at least one of their favorite books.")
    print(f"")
    print(f"Precision @ {number_of_recommendations}: {(final_precision*100):.2f}%")
    print(f"  ↳ Meaning: Out of the {number_of_recommendations} books recommended, an average of {(final_precision*100):.2f}% were correct.")
    print(f"")
    print(f"Recall @ {number_of_recommendations}:    {(final_recall*100):.2f}%")
    print(f"  ↳ Meaning: Out of all the books a user loved, our Top {number_of_recommendations} list caught {(final_recall*100):.2f}% of them.")
    print(f"")
    print(f"NDCG @ {number_of_recommendations}:       {(final_ndcg*100):.2f}%")
    print(f"  ↳ Meaning: Measures ranking quality. High score means correct books appeared at the very top of the list.")
    print(f"="*40)

    return {
        "users_tested": total_users_tested,
        "total_hits": total_hits,
        "hit_ratio": final_hit_ratio,
        "precision": final_precision,
        "recall": final_recall,
        "ndcg": final_ndcg,
        "recommendations_count": number_of_recommendations
    }

if __name__ == "__main__":
    book_pivot, model, final_rating = load_ml_components()
    test_recommendation_system(book_pivot, model, final_rating, number_of_recommendations=5)
