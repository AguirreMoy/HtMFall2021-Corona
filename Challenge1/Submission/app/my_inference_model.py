# THIS IS AN EXAMPLE SCRIPT. YOU CAN USE THIS DIRECTLY OR ADAPT IT TO YOUR NEEDS.

'''
Hackthemachine 2021 Model Inference Script

This is an example of how you might want to organize your model so it can be
easily called by a testing script (such is the case for "my_testing_script.py").

You could also just put all this code into your testing script file.
It is not strictly necessary to split the model into a separate file like this.

Feel free to use this script directly or adapt it as needed for your solution.

What this script does:
1) Load your model
2) Make a prediction on the whole dataset at once
'''
import Ember_Wrapper
# import xgboost as xgb
import numpy as np
import pandas as pd
import time
import pickle

class MyClassifier:

    def __init__(self):
        # good place to load my awesome model here
        # self.model = xgb.Booster()
        self.model = pickle.load(open('xgb_model.pkl', "rb"))

    def predict(self,data):
        data['category'] = 1

        tick = time.time()
        X, y = Ember_Wrapper.create_vectorize_features(data)
        # X = np.load('./X_data.npy')
        tock = time.time()
        print('Time spent for ember wrapper:', tock-tick)

        # X = xgb.DMatrix(X)

        # good place to do whatever data transformations we need
        # ... here we don't need the data b/c we're just guessing the answer
        tick = time.time()
        prediction = self.model.predict(X)
        tock = time.time()
        print('Time spent for inference:', tock-tick)
        prediction = (prediction > 0.354)
        #reference EMBER_roland.ipynb to see how we calculated the threshold


        # good place to have a model.predict() kind of call
        # prediction = random.choice([0,1]) # random guess model
        return(prediction)
