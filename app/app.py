 
import streamlit as st 
from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from keras.preprocessing import image
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings("ignore")

#model = load_model("/Users/hexuser/ECG/app/models/bestCNN2D.h5") 


st.markdown("# Exploratory Visualization of 2D CNN Model")

st.markdown("Representative single lead ECG images belonging to different classes of cardiological conditions")
img=Image.open('images/representative_ECG_images.png')
st.image(img)
st.markdown("The single lead images have been cropped from a more larger 12 lead image and preprocessed for classification purposes. Only lead 2 image is used for classification in this project. There is ongoing effort to use all the lead images as different viewpoints and finally use an ensemble method for classification. The images are input as grayscale to the model after resizing them to a standard dimension for all the classes.")

st.markdown("The images presented are of 4 different classes of cardiological conditions - **Normal,** **Myocardial Infarction (MI),** **Abnormal Heartbeat** and **Previous History of MI**.")

st.markdown("# 2D CNN Model")
#model.summary()
img=Image.open('images/CNN2D_summary.png')
st.image(img,width=500)

from PIL import Image
import requests
from io import BytesIO

st.markdown("# Input an ECG image to the trained model to see the real-time predicted output. The images are prelabelled and randomly selected from the repository.")
st.markdown("Feature TBA: Input your own ECG image")

if st.button("Normal"):
    # img=Image.open("https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/Normal/Cropped_Images/Normal_1Cropped_lead4.png")
    num = np.random.randint(0,100)
    url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/Normal/Cropped_Images/Normal_{}Cropped_lead4.png".format(num)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img, width=700, caption="Normal ECG")

    st.markdown("The model predicted this ECG to be of a person with a **normal** heart.")


if st.button("Myocardial Infarction"):
    num = np.random.randint(0,100)
    url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/MI/Cropped_Images/MI_{}Cropped_lead4.png".format(num)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img, width=700, caption="Myocardial Infarction ECG")

    st.markdown("The model predicted this ECG to be of a person having a **heart attack**.")

if st.button("Abnormal Heartbeat"):
    num = np.random.randint(0,100)
    url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/HB/Cropped_Images/HB_{}Cropped_lead4.png".format(num)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img, width=700, caption="Abnormal Heartbeat ECG")

    st.markdown("The model predicted this ECG to be of a person having a **abnormal heartbeat**.")


if st.button("History of Myocardial Infarction"):
    num = np.random.randint(0,100)
    url = "https://raw.githubusercontent.com/hardikroutray/ECG/main/CroppedECGImages_data_v2/PMI/Cropped_Images/PMI_{}Cropped_lead4.png".format(num)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img, width=700, caption="History of MI ECG")

    st.markdown("The model predicted this ECG to be of a person having a **history of heart attack**.")

st.markdown(
    "The data is publicly available **[here](https://doi.org/10.1016/j.dib.2021.106762)** under Creative Commons License.")
    # images=Image.open('images/meet.png')
    # st.image(images,width=600)
