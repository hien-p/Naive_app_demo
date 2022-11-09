
import streamlit as st
from sklearn.datasets import load_digits

from sklearn.model_selection import cross_val_score



def load_data_digits():
    
    # import data
    digits = load_digits()    
    return digits.data, digits.target

def cross_val_score_model(model, data, target):
    # return accuracy 
    return  cross_val_score(model, data, target, scoring='accuracy', cv=10).mean()

    
