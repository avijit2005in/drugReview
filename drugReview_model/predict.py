import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from drugReview_model import __version__ as _version
from drugReview_model.config.core import config
from drugReview_model.pipeline import drug_review_tfidf_classifier_pipe
from drugReview_model.processing.data_manager import load_pipeline
from drugReview_model.processing.data_manager import pre_pipeline_preparation
from drugReview_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
drugReview_pipe= load_pipeline(file_name=pipeline_file_name)

vectorizer_file_name = f"{config.app_config.vectorizer_save_file}{_version}.pkl"
vectorizer_pipe= load_pipeline(file_name=vectorizer_file_name)

def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=input_data)
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}

    print(validated_data.iloc[0][0])
    review_tokenised = vectorizer_pipe.transform([validated_data.iloc[0][0]])

    predictions = drugReview_pipe.predict(review_tokenised)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    print(config.model_config.condition_mappings)
    if not errors:

        predictions = drugReview_pipe.predict(review_tokenised)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        for key, value in config.model_config.condition_mappings.items():
            if value == predictions[0]:
                print("Condition Detected: {}".format(key))

    return results

if __name__ == "__main__":

    data_in={'review':["My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. We have tried many different medications and so far this is the most effective."],
             'condition':["ADHD"]}
    
    make_prediction(input_data=pd.DataFrame(data_in))
