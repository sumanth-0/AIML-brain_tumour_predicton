import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategoricall.h5')

image=cv2.imread('pred/pred1.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

predict_x=model.predict(input_img) 
result=np.argmax(predict_x,axis=1)


print(result)





