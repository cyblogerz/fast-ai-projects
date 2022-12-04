import gradio as gr
from fastai.vision.all import *

learn = load_learner('./model.pkl')
categories = ('Car in good condition','Damaged Car')

def is_car(x) : return x[0].isupper()

def image_classifier(img):
    pred,index,probs = learn.predict(img) 
    return dict(zip(categories, map(float,probs)))

# image = gr.inputs.Image(shape=(192,192))
image = gr.components.Image(shape=(192,192))
label = gr.components.Label()
examples = ['./car.jpeg','./crash.jpeg','./fcar.jpeg']

intf = gr.Interface(fn= image_classifier,inputs=image,outputs=label,examples=examples)
intf.launch()