# import kfserving
# import numpy as np
# import pandas as pd
# import pickle
# import os
# from typing import List, Dict

# class HeartTransformer(kfserving.KFModel):
#     def __init__(self, name: str, predictor_host: str):
#         super().__init__(name)
#         self.predictor_host = predictor_host
#         # Load the scaler saved during training
#         self.scaler = pickle.load(open("/mnt/models/scaler.pkl", "rb"))

#     def preprocess(self, inputs: Dict) -> Dict:
#         # Get raw data from the request
#         data = inputs["instances"]
#         df = pd.DataFrame(data)
        
#         # Apply scaling
#         scaled_data = self.scaler.transform(df)
        
#         return {"instances": scaled_data.tolist()}

#     def postprocess(self, predictions: Dict) -> Dict:
#         # Convert 0/1 to human-readable labels
#         results = ["High Risk" if p > 0 else "Low Risk" for p in predictions["predictions"]]
#         return {"results": results}

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
#     args, _ = parser.parse_known_args()
    
#     transformer = HeartTransformer("heart-model", predictor_host=args.predictor_host)
#     kfserver = kfserving.KFServer()
#     kfserver.start(models=[transformer])

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import json
import numpy as np
import requests
import kfserving
import argparse
from typing import List, Dict
import logging
import io
import base64
import sys, json
import os
import pandas as pd
from io import StringIO

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument(
    "--model_name",
    default=DEFAULT_MODEL_NAME,
    help="The name that the model is served under.",
)
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)

args, _ = parser.parse_known_args()

class TransformPipeline(kfserving.KFModel):

    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        dirpath = os.path.dirname(os.path.realpath(__file__))
        # self.encoders = pickle.load( open( dirpath + "/encoders.pkl", "rb" ) )

    # def __init__(self):
        # self.model = joblib.load(model_path)
        # self.scaler = StandardScaler()

    def preprocess(self, df):
        """
        Apply same transformations performed in train.py.

        Returns:
            X_scaled: numpy array transformed features
            df: modified dataframe with target col
        """
        # Create binary target column like training
        df["target"] = (df["num"] > 0).astype(int)

        # Feature selection
        X = df.drop(["num", "target"], axis=1)

        # Fit and scale using same approach as train.py
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, df

    def postprocess(self, df, preds):
        """
        Append predictions + log count to MLflow.
        """
        results = ["High Risk" if p > 0 else "Low Risk" for p in preds]
        df["prediction"] = results

        # Simple metric log
        # mlflow.log_metric("pred_count", len(preds))

        return {"results": results}


if __name__ == "__main__":

    # transformer = Transformer(args.model_name, predictor_host=args.predictor_host)
    # kfserver = kfserving.KFServer()
    # kfserver.start(models=[transformer])
    # # Example usage inside a pipeline step
    # pipeline = TransformPipeline("/model/model.joblib")

    # # df = pd.read_csv("/dataset/cleaned_heart_dataset.csv")

    # X_scaled, df_processed = pipeline.preprocess(df)

    # preds = pipeline.model.predict(X_scaled)

    # final = pipeline.postprocess(df_processed, preds)

    # final.to_csv("/output/predictions.csv", index=False)

    # print("Transform step completed.")

    import argparse
    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    args, _ = parser.parse_known_args()
    
    transformer = TransformPipeline(predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])