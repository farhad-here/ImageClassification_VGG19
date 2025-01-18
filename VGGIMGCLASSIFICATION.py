import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import h5py
#style
st.markdown(
    """
       <style>
              .st-emotion-cache-1erivf3{
                     background-color:#25308a;
              }
              .e115fcil1{
                     width:240px;
              }
              .e115fcil1 img{
                     border-radius:1rem ;
              }
              .e115fcil2{
                     display:flex;
                     justify-content: center;
              }
              .stMarkdown .e1nzilvr5 p{
                     color:gold;
                     font-size:20px;
                     text-align:center;
                     font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                     background-color:black;
                     border-radius:13rem ;
                     padding:10px;
                     backdrop-filter: blur(10px);


              }
              .st-emotion-cache-1qg05tj e1y5xkzn3{
                     color:red;
                     font-size:20px;
                     text-align:center;
                     font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
              }
              .e1nzilvr5 p{
                     color:gray;
                     font-size:30px;
              }
              .e1f1d6gn1{
                     height:100;
              }
              .e1f1d6gn4{
                     height:100;
              }
              .e115fcil1{
                     border-radius:10rem;
                     border:5px solid gold;
                     padding: 3rem;
                     margin: 3rem;
                     position: relative;
                     top: 20%;
              }
              .st-emotion-cache-nok2kl h3{
                     width: 100%;
                     color:black;
                     padding: 2rem;
              }
              .e1f1d6gn5 .e1f1d6gn3{
                     background-color: #172d43;
                     
              }
              
              .e1f1d6gn3 , .e1f1d6gn1 , .e1f1d6gn2{
                     display: flex;
                     align-content: center;
                     justify-content: center;
              }
              .e1f1d6gn1{
                     display: flex;
                     align-content: center;
                     justify-content: center;
              }
              .e1f1d6gn2{
                     display: flex;
                     align-content: center;
                     justify-content: center;
              }
              .e1f1d6gn0{
                     display: flex;
                     justify-content: center;
                     align-content: center;
              }
              .st-emotion-cache-nok2kl{
                     display:flex;
                     flex-direction:column;
                     justify-content:center;
                     align-content:center;
                     background-color:gray;
                     border:5px solid gold;
              }
              
       </style>
       <body>
              <p style='color:gold;border:2px solid white;padding:1rem;margin:1rem;font-size:40px;font-weight:bold;background-color:darkgray;'>Image Classification</p>
              <p style='color:seagreen;font-size:20px;padding:1rem;margin:1rem'>In my github there is a script for image classifications, after putting dataset in it, the model.h5 file would be save then put that file here</p>
              <p style='padding:1rem;margin:1rem'>
                     Made by FarhadGhaherdoost
              </p>
              <a style='text-align:center;padding:1rem;margin:1rem; text-decoration: none;color:black;font-size:20px; border-radius:30rem;background-color:#00c3ff;' href=' https://github.com/farhad-here'>My github</a>
       </body>
    """, unsafe_allow_html=True
)

#========================================== Upload ==========================================
# Upload the model file
model_file = st.file_uploader('Upload Model (Model.h5)', type='h5')
#warning
col1,col2 = st.columns(spec=2,gap='large')
col1.info('### !The image must be like this \n ### !Absolute white background with no shodow \n ### !This will result to predict better with more accuracy')
col2.image('./hammer (13).jpg')
# Upload the image file
pic = st.file_uploader('Upload Image for Prediction', type=['jpg', 'png', 'jpeg'])

# Make sure the model and image files are uploaded
if model_file is not None and pic is not None:
       # Load the model from the uploaded file
       loaded_model = load_model(model_file.name)

       # Open the uploaded image
       image = Image.open(pic)

       # Preprocess the image: resize to (224, 224), convert to RGB, convert to numpy array, and normalize
       img = image.resize((224, 224))  # Resize to match model's expected input
       img = img.convert('RGB')  # Ensure 3 channels (RGB)
       img_array = np.array(img).astype('float32')  # Convert to numpy array and cast to float32
       img_array = img_array / 255.0  # Rescale to [0, 1]

       # Add a batch dimension
       img_array = np.expand_dims(img_array, axis=0)  # Shape should now be (1, 224, 224, 3)

       # Make prediction using the loaded model
       prediction = loaded_model.predict(img_array)

       # Get the class with the highest probability
       predicted_class = np.argmax(prediction)
       try:
              with h5py.File(f'{model_file.name}', 'r') as f:
                     class_names = list(f['class_names'])  # Retrieve class names
                     class_names = [name.decode('utf-8') for name in class_names]  # Decode from bytes to strings
              # Print the class names
              # If you have class labels, print them, for example:
              st.write(f'Predicted class: {class_names[predicted_class]}')
              st.image(img)
       except:
              st.write(f'Predicted class: {predicted_class}')
              st.image(img)