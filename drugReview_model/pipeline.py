import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline

from drugReview_model.config.core import config
from drugReview_model.processing.features import Mapper
import xgboost as xgb

drug_review_tfidf_classifier_pipe=Pipeline([

     ##==========Mapper======##
     #("map_label", Mapper(config.model_config.condition_var, config.model_config.condition_mappings)
     # ),
     ('model_xgb', xgb.XGBClassifier(config.model_config.objective, config.model_config.label_count,
                                     config.model_config.nthread, config.model_config.tree_method, config.model_config.device))
     ])