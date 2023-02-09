# Import Libraries

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from pathlib import Path
import sys
from streamlit.web import cli as stcli
from streamlit import runtime
from utils import load_model
from DataManipulation import DataManipulation
import warnings
warnings.filterwarnings("ignore")

#set to wide mode  
st.set_page_config(layout="wide")

predictor = load_model('assets/lgbm_customer_churn.pkl')

#user input - Policy No
def customerNo():
    customerNo = st.sidebar.number_input('Customer ID', min_value=0,max_value=999999999999999, step=1)
    st.write("customerid : " + str(customerNo))
    return str(customerNo)

def userInputFeatures():
    st.sidebar.write('**Customer Data**')
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    seniorcitizen = st.sidebar.selectbox('Is Senior', ('Yes', 'No'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))

    st.sidebar.markdown("""---""")
    st.sidebar.write('**Product & Service**')
    tenure = st.sidebar.slider('Tenure', 0, 100)
    phoneservice = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    multiplelines = st.sidebar.selectbox('Multiplelines', ('Yes', 'No'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paymentmethod = st.sidebar.selectbox('Payment method', ('Electronic Check', 'Mailed Check', 'Bank Transfer (automatic)', 'Credit Card (automatic)'))
    monthlycharges = st.sidebar.number_input('Monthly Charges')


    data = {'gender' : gender.lower(), 'seniorcitizen': seniorcitizen.lower(), 'partner': partner.lower(), 'tenure': tenure, 'phoneservice': phoneservice.lower(),
    'multiplelines': multiplelines.lower(), 'contract': contract.lower(), 'paymentmethod': paymentmethod.lower(), 'monthlycharges': monthlycharges}

    return pd.DataFrame(data, index=[0])    


#export prediction result to a excel file
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def main():
    #header
    st.write('# Customer Churn Prediction')

    #header of the sidebar
    st.sidebar.header('User Input')

    #import file or user manually inputs
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        if Path(uploaded_file.name).suffix == '.xlsx':
            df = pd.read_excel(uploaded_file, index_col=False)
        else:
            df = pd.read_csv(uploaded_file, index_col=False)

        #clean data
        df = DataManipulation.tidy_data(df)
        

    else:
        cust_no = customerNo()
        df = userInputFeatures()  
        df = pd.concat([pd.DataFrame({'customerno': cust_no}, index=[0]),df], axis=1)

    df.reset_index(drop=True, inplace=True)
    #display dataframe without data manipulation
    display_df = df.copy()

    #data manipulation
    df['gender'] = DataManipulation.encode_gender(df['gender'])
    df['seniorcitizen'] = DataManipulation.encode_seniorcitizen(df['seniorcitizen'])
    df['partner'] = DataManipulation.encode_partner(df['partner'])
    df['phoneservice'] = DataManipulation.encode_phoneservice(df['phoneservice'])
    df['multiplelines'] = DataManipulation.encode_multiplelines(df['multiplelines'])
    df['contract'] = DataManipulation.encode_contract(df['contract'])
    df['paymentmethod'] = DataManipulation.encode_paymentmethod(df['paymentmethod'])
    df['monthlycharges'] = DataManipulation.scale_monthlycharges(df[['monthlycharges']])


    #displays the user input features
    st.subheader('User Input')
    
    if uploaded_file is not None:
        display_df = pd.concat([display_df], axis=1)
        st.dataframe(display_df,1500,500)
    else:
        st.table(display_df)

    #prediction
    prediction = predictor.predict(df.drop(columns=['customerno']))
    prediction_proba = predictor.predict_proba(df.drop(columns=['customerno']))

    prediction_df = pd.DataFrame({'Probability(%)': prediction_proba[:, 1] * 100, 'Churn': ['Yes' if x > 0.5 else 'No' for x in prediction]})
    st.markdown('***---***')
    st.markdown('Prediction')
    st.markdown('##### Notes')
    st.markdown('######  Label as Churn if Probability(%) more than or equal to 50%' )

    
    try:
        df2 = pd.concat([df['customerno'], prediction_df], axis=1)
    except UnboundLocalError:
        df2 = pd.concat([prediction_df], axis=1)

    

    df2.reset_index(drop=True)
    #df2 = df2.sort_values(by=['customerno'], ascending=False)
    st.dataframe(df2.style.format({'Probability(%)': '{:.2f}'}))

    try:
        df2['customerno'] = df2['Customer No'].astype(np.int64)
        df2['Probability(%)'] = df2['Probability(%)'].round(decimals=2)
        df2_xlsx = to_excel(df2)
        st.download_button(label='Export Current Result',
                           data=df2_xlsx,
                           file_name=f'{Path(uploaded_file.name).stem}_result.xlsx')
    except KeyError:
        df2['Probability(%)'] = df2['Probability(%)'].round(decimals=2)
        
        df2_xlsx = to_excel(df2)
        st.download_button(label='Export Current Result',
                           data=df2_xlsx,
                           file_name=f'result.xlsx')

if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ['streamlit', 'run', 'app.py']
        sys.exit(stcli.main())


