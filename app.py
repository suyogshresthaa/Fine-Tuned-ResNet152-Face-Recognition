# This is the Streamlit app for Cristiano Ronaldo Face Classifier



##################### References #############################
"""
Github example 1: https://github.com/aleahy-work/CS-STAT323-W26/blob/main/ClassJupyter/streamlit-test.py
Github example 2: https://github.com/aleahy-work/CS-STAT323-W26/blob/main/ClassJupyter/streamlit-ollama1.py
"""
##############################################################



##################### Imports #############################

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

##################### Loading the Model #############################

@st.cache_resource           # prevent model from reloading every time
def load_model():
    """
    We load the pretrained ResNet152 model with the weigths we saved. For this, we need to recreate the same model architecture used during training.
    """
    model = models.resnet152(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))      # loading the best weigths

    model.eval()    # set to evaluation mode
    return model

##################### Preprocessing the Image #########################

def transform_image(image):
    """
    We need to apply the same transforms that we used during validation to preprocess our image.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(image).unsqueeze(0)

##################### Model Prediction #########################

def predict(model, image_t):
    """
    We need to perform inference and return the predicted class
    """
    class_names = ['Not_Ronaldo', 'Ronaldo']

    with torch.no_grad():
        outputs = model(image_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)    # converting outputs to probs
        confidence, pred = torch.max(probabilities, 1)

    pred_class = pred.item()      # predicted class 
    confidence_score = confidence.item() * 100      # how confident the model is in its prediction
    
    return class_names[pred_class], confidence_score      # returning class label

##################### App Layout #########################

st.set_page_config(page_title='Cristiano Ronaldo Image Classifier')      # page title

st.title("Cristiano Ronaldo Face Classifier Using PyTorch's RestNet152 Model")    

st.write("""
This app uses a fine-tuned **ResNet152** CNN model to determine whether 
an uploaded facial image is of **Cristiano Ronaldo** or not. The model was trained 
using transfer learning on a dataset of facial images and achieved a validation 
accuracy of **96.63%**. Simply upload an image and click the button to find out!
""")

# information 
st.info("""
**For best results:**
- Use a clear frontal face image.
- Make sure the face is clearly visible and well lit.
- Avoid group photos.
- Supported formats: JPG, PNG, JPEG
""")

# model information bar on the side
with st.sidebar:
    st.header('About the Model')
    st.write("""
    **Architecture:** ResNet152
    
    **Training Method:** Transfer Learning
    
    **Dataset:**
    - Ronaldo images: 109
    - Not Ronaldo images: 332
    - Total: 441 images
    
    **Training Split:** 80% train, 20% validation
    
    **Best Validation Accuracy:** 96.63%
    
    **Classes:**
    - Ronaldo
    - Not Ronaldo
    """)

model = load_model()

uploaded_file = st.file_uploader('Please choose an image to upload', type=['jpg', 'png', 'jpeg'])      # uploader box

if uploaded_file is not None:
    # displaying the image
    image = Image.open(uploaded_file).convert('RGB')    # ensure RGB
    col1, col2, col3 = st.columns([1,2,1])  # creating 3 columns so the image can go in the center
    with col2:
        st.image(image, caption='Uploaded Image', width=300)

    st.write('')

    if st.button('Click to Find Out the Prediction'):
        # transforming image and running prediction
        with st.spinner('Classifying...'):
            image_t = transform_image(image)    # image tensor
            result, confidence = predict(model, image_t)

        # displaying result
        st.write('')
        st.write('Our prediction is...')
        if result == 'Ronaldo':
            st.success('This is Cristiano Ronaldo. SUIIIIIIIII!!!')
            st.metric(label='Confidence Score', value=f'{confidence:.2f}%')
        else:
            st.error('This is not Cristiano Ronaldo.')
            st.metric(label='Confidence Score', value=f'{confidence:.2f}%')
