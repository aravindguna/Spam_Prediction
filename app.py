#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle


# load the model from disk

# In[2]:


filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)


# In[3]:


@app.route('/')
def home():
    return render_template('home.html')


# In[4]:


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


# In[ ]:


if __name__ == '__main__':
    app.run(debug=False)


# In[ ]:




