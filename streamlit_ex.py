# import the streamlit library
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle

model_filename = 'crop_prediction_model.sav'

loaded_model = pickle.load(open(model_filename, 'rb'))


# give a title to our app
st.title('Welcome to AI Based Optimal Crop prediction')

# TAKE Nitrogen INPUT in kgs
N = st.number_input("Enter nitrogen content of soil")



# TAKE Phosporous
P = st.number_input("Enter ratio of Phosphorous content in soil")

K = st.number_input("Enter ratio of Potassium content in soil")

# TAKE Temperature
temp = st.number_input("Enter temperature of the place in °C")

# TAKE Humidity
humid = st.number_input("Enter relative humidity in '%' of the place")

# TAKE ph
ph = st.number_input("Enter ph of the soil (0-14 scale only)")

# TAKE rainfall
rain = st.number_input("Enter estimated rainfall in mm")

sample = pd.DataFrame([[N,P,K,temp,humid,ph,rain]],columns = ['N','P','K','temperature','humidity','ph','rainfall'])


targets = {0: 'apple',
 1: 'banana',
 2: 'blackgram',
 3: 'chickpea',
 4: 'coconut',
 5: 'coffee',
 6: 'cotton',
 7: 'grapes',
 8: 'jute',
 9: 'kidneybeans', #rajma
 10: 'lentil', #vaal
 11: 'maize', #corn 
 12: 'mango',
 13: 'mothbeans', #beans 
 14: 'mungbean', 
 15: 'muskmelon',
 16: 'orange',
 17: 'papaya',
 18: 'pigeonpeas', #whole toor
 19: 'pomegranate',
 20: 'rice',
 21: 'watermelon'}


images = {
    0:'https://images.unsplash.com/photo-1576179635662-9d1983e97e1e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=987&q=80',
    1:'https://images.unsplash.com/photo-1603833665858-e61d17a86224?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NXx8YmFuYW5hfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',
    2:'https://media.istockphoto.com/id/1199245897/photo/urad-dal-black-gram-vigna-mungo-on-white-background.jpg?b=1&s=170667a&w=0&k=20&c=gViJZR14iH0mGXShZCyfc9ztwIiN_0IrTYzDSrBPa0E=',
    3:'https://images.unsplash.com/photo-1515543904379-3d757afe72e4?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8Y2hpY2twZWF8ZW58MHx8MHx8&auto=format&fit=crop&w=800&q=60',
    4:'https://images.unsplash.com/photo-1580984969071-a8da5656c2fb?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTF8fGNvY29udXR8ZW58MHx8MHx8&auto=format&fit=crop&w=800&q=60',
    5:'https://images.unsplash.com/photo-1511920170033-f8396924c348?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NHx8Y29mZmVlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',
    6:'https://images.unsplash.com/photo-1634337781106-4c6a12b820a1?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MjB8fGNvdHRvbnxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=800&q=60',
    7:'https://images.unsplash.com/photo-1602330102257-04c00af50c1a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTV8fGdyYXBlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=800&q=60',
    8:'https://images.unsplash.com/photo-1664688708942-c77a6b5e6abc?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8anV0ZSUyMHBsYW50fGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',
    9:'https://images.unsplash.com/flagged/photo-1577226259316-c566059cd6fc?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NHx8a2lkbmV5JTIwYmVhbnN8ZW58MHx8MHx8&auto=format&fit=crop&w=800&q=60',
    10:'https://images.unsplash.com/photo-1614373532201-c40b993f0013?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8bGVudGlsc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=800&q=60',
    11:'https://images.unsplash.com/photo-1649251037465-72c9d378acb6?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NTR8fG1haXplfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',
    12:'https://images.unsplash.com/photo-1601493700631-2b16ec4b4716?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8bWFuZ298ZW58MHx8MHx8&auto=format&fit=crop&w=800&q=60',
    13:'https://t2.gstatic.com/licensed-image?q=tbn:ANd9GcT9xIxsV0QkDRur9ojYJ4napcWcxcCBnSzDE6nj27c96EhLIjgE8L3CQrxJcXWNHNdTFPsu-MgqbYjUdlY',
    14:'http://t2.gstatic.com/licensed-image?q=tbn:ANd9GcQY3QYV3GnlkM-7mRDagGmVo3ADifW9anvxtGJ9RbV7dUWNHe716X_r8cKYMxLxORm6FGvotlaazsNwnXw',
    15:'https://images.unsplash.com/photo-1602597190461-43774583d3c0?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8bXVzayUyMG1lbG9ufGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',
    16:'https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8OHx8b3JhbmdlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',
    17:'https://images.unsplash.com/photo-1623492229905-ebc1202e8904?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NXx8cGFwYXlhfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',
    18:'http://t3.gstatic.com/licensed-image?q=tbn:ANd9GcQ-WwJwMnCRCqMGqKYWLzBXYcge9l9VJFl2QEaMNuuTtVlE90nV6aXt0RWOG86w90qXNZpQvzXPNwnBX_Q',
    19:'https://images.unsplash.com/photo-1541344999736-83eca272f6fc?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MjB8fHBvbWVncmFuYXRlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60',    
    20:'https://images.unsplash.com/photo-1586201375761-83865001e31c?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2370&q=80',
    21:'https://images.unsplash.com/photo-1621583441131-c8c190794970?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1364&q=80',



}
def getPrediction(x):
    return targets.get(x[0],'Couldnt find any :(. Please check the inputs'),images.get(x[0],'Couldnt find any :(. Please check the inputs')

# check if the button is pressed or not
if(st.button('Predict Optimal Crop')):
    pred,img = getPrediction(loaded_model.predict(sample))
    if pred == 'Couldnt find any :(. Please check the inputs':
        st.text("Couldnt find any optimal crop :(. Please check the inputs and try again")
    else:
        st.image(img, caption=pred)
        st.text("The most optimal crop given the scenario is {}.".format(pred))
        