import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from PIL import Image
import PIL.ImageOps
import os
import ssl

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context' , None)) : 
    # context is a setting used for creating a secured connection with the website.
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image_p.npz')['arr_0']
y = pd.read_csv('pro_labels.csv')['labels']

print(pd.Series(y).value_counts())

classes = ['A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G' , 'H' , 'I' , 'J', 'H' , 'I' , 'J' , 'K' , 'L' , 'M' , 'N' , 'O' , 'P' , 'Q' , 'R' , 'S' , 'T' , 'U' , 'V' , 'W' , 'X' , 'Y' , 'Z']
n_classes = len(classes)

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state = 9 , train_size = 7500 , test_size = 2500)

clf = LogisticRegression(solver='saga' , multi_class='multinomial').fit(X_train , y_train)

def getPrediction() :
    im_pil = Image.open(image)

    image_bw = im_pil.convert('L') 

    image_bw_resized = image_bw.resize((22,30) , Image.ANTIALIAS)

    pixel_filter = 20 

    min_pixel = np.percentile(image_bw_resized, pixel_filter) 
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255) 

    max_pixel = np.max(image_bw_resized) 

    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 

    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)

    prediction = clf.predict(test_sample)
    return(prediction[0])
