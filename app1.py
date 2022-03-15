import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='wide', initial_sidebar_state="collapsed")

  
# loading in the model to predict on the data
pickle_in = open('LogisticRegression.pkl','rb')
classifier = pickle.load(pickle_in)


original_title = '<p style="font-family:sans-serif; color:Green; font-size: 20px;">Welcome to the app! ꒱࿐♡ ˚.*ೃ</p>'
st.markdown(original_title, unsafe_allow_html=True)


st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon. Check the source code [here](https://github.com/asma0228/crop-recommendation/blob/main/app1.py)")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
  
def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs



def prediction(date, crop_type, crop_sown, rainfall, temp_avg,
       humidity_avg, wind_speed_avg, mrp, Min_duration, max_duration,
       duration, avg_cost_of_cultivation, Yield, Soil_type, avg_pH,
       N, P, K, irrigation, gross_profit, net_profit, ROI,
       sow_and_harvest, crop_term):  
   
    prediction = classifier.predict(
        [[date, crop_type, crop_sown, rainfall, temp_avg,
               humidity_avg, wind_speed_avg, mrp, Min_duration, max_duration,
               duration, avg_cost_of_cultivation, Yield, Soil_type, avg_pH,
               N, P, K, irrigation, gross_profit, net_profit, ROI,
               sow_and_harvest, crop_term]])
    print(prediction)
    return prediction
   
   
  
# this is the main function in which we define our webpage 
def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2, col3, col4  = st.columns([2,2,2,2])
    
    with col1: 
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''
    
    with col2:
     date = st.text_input("Type date", "Type Here")
     crop_sown = st.text_input("Crop sown", "Type Here")
     rainfall = st.text_input("Rainfall (mm)", "Type Here")
     temp_avg = st.text_input("Average Temp (°C)", "Type Here")
     humidity_avg = st.text_input("Average humidity (%)", "Type Here")
     wind_speed_avg = st.text_input("Average wind speed (Kmph)", "Type Here")
     mrp = st.text_input("Maximum retail prices (MRP rs/kg)", "Type Here")
     Min_duration = st.text_input("Minimum duration (days)", "Type Here")
     max_duration = st.text_input("Maximum duration (days)", "Type Here")
     duration = st.text_input("Duration (months)", "Type Here")
     avg_cost_of_cultivation = st.text_input("Avg cost of cultivation (rs/ac)", "Type Here")
     Yield = st.text_input("Yield (kg/ac)", "Type Here")
     avg_pH = st.text_input("Average PH of soil", "Type Here")
     
    with col3: 
     N = st.text_input("N (kg/ha)", "Type Here")
     P = st.text_input("P (kg/ha)", "Type Here")
     K = st.text_input("K (kg/ha)", "Type Here")
     gross_profit = st.text_input("Gross Profit (rs/ac)", "Type Here")
     net_profit = st.text_input("Net Profit (rs/ac)", "Type Here")
     ROI = st.text_input("Return on investment (%)", "Type Here")
     st.write(f"Soil type")
     checkbox_30 = st.checkbox("loamy soil")
     checkbox_31 = st.checkbox("sandy loam")
     checkbox_32 = st.checkbox("rich loam")
     checkbox_33 = st.checkbox("deep loam")
     checkbox_34 = st.checkbox("sandy soil")
     checkbox_35 = st.checkbox("light loamy")
     checkbox_36 = st.checkbox("fertile loam")
     checkbox_37 = st.checkbox("red sandy loam")
     checkbox_38 = st.checkbox("clay loam")
     checkbox_39 = st.checkbox("deep black")
     checkbox_40 = st.checkbox("clay")
 


     if checkbox_30:
            Soil_type = 0
     elif checkbox_31:
            Soil_type = 1
     elif checkbox_32:
            Soil_type = 2
     elif checkbox_33:
            Soil_type = 3
     elif checkbox_34:
            Soil_type = 4
     elif checkbox_35:
            Soil_type = 5
     elif checkbox_36:
            Soil_type = 6
     elif checkbox_37:
            Soil_type = 7
     elif checkbox_38:
            Soil_type = 8
     elif checkbox_39:
            Soil_type = 9
     elif checkbox_40:
            Soil_type = 10

    with col4:
    
     st.write(f"Crop type")
     checkbox_41 = st.checkbox("grains")
     checkbox_42 = st.checkbox("fruit")
     checkbox_43 = st.checkbox("flower")
     checkbox_44 = st.checkbox("oil_seeds")
     checkbox_45 = st.checkbox("other_commercial_crop")
     checkbox_46 = st.checkbox("vegetable")
     if checkbox_41:
        crop_type = 0
     elif checkbox_42:
        crop_type = 1
     elif checkbox_43:
        crop_type = 2
     elif checkbox_44:
        crop_type = 3
     elif checkbox_45:
        crop_type = 4
     elif checkbox_46:
        crop_type = 5
      
            
     st.write(f"Irrigation")
     checkbox_47 = st.checkbox("yes")
     checkbox_48 = st.checkbox("no")
     if checkbox_47:  
      irrigation = 0      
     elif checkbox_48:     
      irrigation = 1
    
             
                
     st.write(f"Sow and harvest")
     checkbox_49 = st.checkbox("one_sow_one_harvest")
     checkbox_50 = st.checkbox("one_sow_few_harvests")
     checkbox_51 = st.checkbox("one_sow_many_harvests")
     if checkbox_49:
      sow_and_harvest = 0
     elif checkbox_50:
      sow_and_harvest = 1
     elif checkbox_51:
      sow_and_harvest = 2
                  
                  
     st.write(f"Crop term")
     checkbox_52 = st.checkbox("short_term")
     checkbox_53 = st.checkbox("intermediate_term")
     checkbox_54 = st.checkbox("long_term")
     if checkbox_52:
      crop_term = 0
     elif checkbox_53:
      crop_term = 1
     elif checkbox_54:
      crop_term = 2
         

    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    with col4:
     if st.button("Predict 🔍 "):
       result = prediction(date, crop_type, crop_sown, rainfall, temp_avg,humidity_avg, wind_speed_avg, mrp, Min_duration, max_duration,duration, avg_cost_of_cultivation, Yield, Soil_type, avg_pH,N, P, K, irrigation, gross_profit, net_profit, ROI,sow_and_harvest, crop_term )
    col4.success('🌻 The best crop recommended by the A.I for your farm is 🢂{}'.format(result))



if __name__=='__main__':
    main()
    
