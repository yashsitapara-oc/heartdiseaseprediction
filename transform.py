import kfserving
import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict

class HeartTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        # Load the scaler saved during training
        self.scaler = pickle.load(open("/mnt/models/scaler.pkl", "rb"))

    def preprocess(self, inputs: Dict) -> Dict:
        # Get raw data from the request
        data = inputs["instances"]
        df = pd.DataFrame(data)
        
        # Apply scaling
        scaled_data = self.scaler.transform(df)
        
        return {"instances": scaled_data.tolist()}

    def postprocess(self, predictions: Dict) -> Dict:
        # Convert 0/1 to human-readable labels
        results = ["High Risk" if p > 0 else "Low Risk" for p in predictions["predictions"]]
        return {"results": results}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    args, _ = parser.parse_known_args()
    
    transformer = HeartTransformer("heart-model", predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])