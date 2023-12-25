import streamlit as st
import pandas as pd
import numpy as np

def predict_cii(deadweight):
    return 1984 * deadweight**(-0.489)

def calculate_future_cii(deadweight,utilization,distance, start_year, end_year):
    years = range(start_year, end_year + 1)
    reduction_factors = [0, 2, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

    data = {'Year': [], 'CII': [], 
            'Improved_CII': [], 'Payload': [],
            'Transport_Work': [],'CO2Emission': [],'HFO_CONS':[]}

    for i, year in enumerate(years):
        cii = predict_cii(deadweight)
        reduction_factor = reduction_factors[i] / 100
        improved_cii = cii * (1 - reduction_factor)
        reference_payload = deadweight * utilization / 100
        transportwork = distance * deadweight
        emission =  transportwork * improved_cii / 10**6
        cons = emission / 3.114
        
        data['Year'].append(year)
        data['CII'].append(cii)
        data['Improved_CII'].append(improved_cii)
        data['Payload'].append(reference_payload)
        data['Transport_Work'].append(transportwork)
        data['CO2Emission'].append(emission)
        data['HFO_CONS'].append(cons)
        

    df = pd.DataFrame(data)
    return df

def main():
    st.title("CII Value Predictor")

    # Get input from the user
    deadweight = st.number_input("Enter Deadweight", min_value=5000.0)
    utilization = st.number_input("Enter Utilization (%)", min_value=0.0, max_value=100.0, value=100.0)
    distance = st.number_input("Enter Distance", min_value=1.0)
    start_year = 2020
    end_year = 2030

    # Predict CII value for the input year
    cii_prediction = predict_cii(deadweight)
    st.write(f"Refernce CII Value in 2019: {cii_prediction:.2f}")

    # Calculate future CII values
    df = calculate_future_cii(deadweight,utilization,distance, start_year, end_year)

    # Display the DataFrame
    st.dataframe(df)

    # Display the chart
    st.line_chart(df.set_index('Year')[['CII', 'Improved_CII']], use_container_width=True)

if __name__ == "__main__":
    main()
