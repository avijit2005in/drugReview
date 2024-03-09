import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from drugReview_model.config.core import config
from drugReview_model.pipeline import drug_review_tfidf_classifier_pipe
from drugReview_model.processing.data_manager import load_dataset, save_pipeline, save_vectorizer
from sklearn.metrics import f1_score, accuracy_score

def run_training() -> None:
    
    """
    Train the model.
    """
    # read training data
    data = load_dataset(file_name=config.app_config.drugs_data_file)

    tfidf_vectorizer = TfidfVectorizer(max_features=config.model_config.max_features)
    # Fit the vectorizer on the encoded category data
    tfidf_review = tfidf_vectorizer.fit_transform(data[config.model_config.features[0]])

    # Convert the sparse matrix to a dense array
    tfidf_review = tfidf_review.toarray()

    save_vectorizer(data_to_persist=tfidf_vectorizer)

    # Separate features and target
    X = tfidf_review
    Y = data[config.model_config.target]

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    drug_review_tfidf_classifier_pipe.fit(X_train,y_train)
    # Make predictions on the test data
    predictions = drug_review_tfidf_classifier_pipe.predict(X_test)
    # Evaluate the model
    f1 = f1_score(y_test, predictions, average='macro')
    accuracy = accuracy_score(y_test, predictions)
    print("F1 Score:", f1)
    print("Accuracy Score:", accuracy)

    # persist trained model
    save_pipeline(pipeline_to_persist= drug_review_tfidf_classifier_pipe)
    # printing the score
    
if __name__ == "__main__":
    run_training()