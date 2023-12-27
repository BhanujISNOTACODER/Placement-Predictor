import pickle
import pandas as pd
import streamlit as st
st.title("Placement Predictor: ")

IQ = st.number_input('Enter IQ of the student',min_value=40.00,max_value=220.00,format="%.2f")
CGPA = st.number_input('Enter CGPA of the student',min_value=3.00,max_value=10.00,format="%.2f")


predictor = pickle.load(open('../exports/model.pkl','rb'))
scaler = pickle.load(open('../exports/scalar.pkl','rb'))

inp_dict = {
    "cgpa":CGPA,
    "iq":IQ
}
df = pd.DataFrame([inp_dict])
newData = scaler.transform(df)
res = predictor.predict(newData)


if st.button('Predict'):
    if(res[0]==0):
        st.write('Not Placed')
    else:
        st.write("Placed")