# THIS IS AN EXAMPLE SCRIPT. YOU CAN USE THIS DIRECTLY OR ADAPT IT TO YOUR NEEDS.

'''
Hackthemacine 2021 Model Inference Script

This is an example of how you might want to organize your model so it can be
easily called by a testing script (such is the case for "my_testing_script.py").

You could also just put all this code into your testing script file.
It is not strictly necessary to split the model into a separate file like this.

Feel free to use this script directly or adapt it as needed for your solution.

What this script does:
1) Load your model
2) Make a prediction on a single piece of data
'''


import pickle
import random

class MyClassifier:

    def __init__(self):
        # good place to load my awesome model here

    def predict(self,data):

        # good place to do whatever data transformations we need
        # ... here we don't need the data b/c we're just guessing the answer

        # good place to have a model.predict() kind of call
        prediction = random.choice([0,1]) # random guess model
        return(prediction)
