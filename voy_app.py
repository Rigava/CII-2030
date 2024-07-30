import requests
import pandas as pd
import io
import streamlit as st

st.set_page_config(page_title="Voyage Intensity Dashboard", page_icon=":ship:", layout="wide")
st.title(" :bar_chart: Impact of EU regulation on Transportation cost")
st.markdown("_Prototype v0.1.0_")
st.write("""EU aims to reduce GHG emissions by minimum 55% by 2030 as compared to 1990 levels.
         In this analysis we have explored the cost implication on shipper due to EU regulations.To promote decarbonization of fuels used onboard ships, 
         EU ETS was extended to shipping sector in 2024 and soon in 2025 FUEL EU Maritime will penalize the vessels if GHG intensity exceeds the 89.34 gCO2eq/MJ
         For more details on the FUEL EU you can read this amazing FAQ documnet here- https://transport.ec.europa.eu/transport-modes/maritime/decarbonising-maritime-transport-fueleu-maritime/questions-and-answers-regulation-eu-20231805-use-renewable-and-low-carbon-fuels-maritime-transport_en)
         """)
with st.expander(" ♻️ Insights"):
         st.markdown("""
         <style>
         [data-testid="stMarkdownContainer"] ul {
             padding-left: 40px;
         }
         </style>
         The data is based on roughly 4400 historical voyage data which was publically available. The calculation is based on consumption of VLSFO fuel type only.
         Assumptions used to calculate the EU penalty:  
         - GHG Intensity for VLSFO 91.7 gCO2eq/MJ
         - EU Allowance is fixed @ 70 Euros and 70 % of fuel consumption accounted for year 2025
         
         A voyage from Rotterdam to Algeciras on anaverage will become expensive by 20 euros per nautical mile due to penalty on fossil fuel imposed by
         upcoming FUEL EU regulation.
         Overall Asia Europe services will attract a surcharge cost of 70 to 80 euros based on the final destination.
         """, unsafe_allow_html=True)



url = "https://raw.githubusercontent.com/Rigava/DataRepo/main/voyage_data.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8'))) 
print(df.info())
df.columns = df.columns.str.strip()
# st.write(df.columns)
#Filtered Dataframe as per trade cluster
df_fil = df[df["Trade Cluster"]=='EMA - Europe Middle East Asia']
df_fil =    df_fil.drop(['Trade Cluster','Segment Size','Alongside (UTC)','ME Hrs BerthToBerth','Nominal TEU','Arrival Waiting Hours','Departure (UTC)'],axis=1)
# Get a list of unique ports from the original DataFrame
filter_toports = df_fil['To Port Name'].unique()
# Create a multi-select widget to select arr ports
fil_ports = st.sidebar.multiselect('Choose the arrival ports',filter_toports,default= "ALGECIRAS")
#Transformation of data fram filtered by arrival ports
df_selected = df_fil[df_fil["To Port Name"].isin(fil_ports)]
df1 = df_selected.groupby('Vessel ID')[['Consumption (MT)','Sea Distance','Sea ME Hours']].sum()
#Assuming all fuel type in HFO
df1['emission_ttw'] = df1['Consumption (MT)'] * 3.125
#Assuming 70 euro is the cost of EUA in 2026 and 70 % of fuel consumption accounted for
df1['EU_ETS_cost'] = df1['emission_ttw'] * 70 *0.7
#Assuming 50% of consumption is to be considered and .0402 MJ/gm
df1['energy_used'] = (df1['Consumption (MT)'] *1000000 *0.0402) 
#Assuming GHG intensity of 91.7 for HFO against the required value of 89.34(2% lower than GHG intensity of 2020 i.e. 91.16)
df1['compliance_deficit'] = (89.34-91.7)*df1['energy_used']
df1['EU_penalty']=(df1['compliance_deficit']*2400)/(91.7 * 41000)
df1['total_penalty'] =df1['EU_ETS_cost']-df1['EU_penalty']
#Metrics
# total_teu =df1['Actual Teu'].sum()
total_distance = df1['Sea Distance'].sum()
total_time = df1['Sea ME Hours'].sum()
total_penalty =df1['total_penalty'].sum()
eu_penalty = abs(df1['EU_penalty'].sum())
eu_ets = df1['EU_ETS_cost'].sum()
eu_ets_pernm =eu_ets / df1['Sea Distance'].sum()
eu_penalty_pernm = eu_penalty / df1['Sea Distance'].sum()
total_eu_cost_pernm = df1['total_penalty'].sum() / df1['Sea Distance'].sum()
total_speed = df1['Sea Distance'].sum()/df1['Sea ME Hours'].sum()

print(eu_ets_pernm)

with st.expander("🔍 Data Preview"):
    st.dataframe(
        df_selected,
        column_config={"Actual Teu": st.column_config.NumberColumn(format="%d"),
                       "Vessel ID": st.column_config.NumberColumn(format="%d"),
                       "total_penalty": st.column_config.NumberColumn(format="%d")},
        
    )
import plotly.express as px
import plotly.graph_objects as go
import random
#######################################
# VISUALIZATION METHODS
#######################################


def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 22},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 15,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 20},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)

#######################################
# STREAMLIT LAYOUT
#######################################

top_left_column, top_right_column = st.columns((2, 1))
bottom_left_column, bottom_right_column = st.columns(2)

with top_left_column:
    column_1, column_2, column_3, column_4 = st.columns(4)

    with column_1:
        plot_metric(
            "EU ETS cost",
            eu_ets,
            prefix="$",
            suffix="",
            show_graph=True,
            color_graph="rgba(0, 104, 201, 0.2)",
        )
        plot_gauge(eu_ets_pernm, "#0068C9", "euros", "EU ETS per NM", 100)

    with column_2:
        plot_metric(
            "Fuel EU Penalty",
            eu_penalty,
            prefix="$",
            suffix="",
            show_graph=True,
            color_graph="rgba(255, 43, 43, 0.2)",
        )
        plot_gauge(eu_penalty_pernm, "#FF8700", " euros", "Fuel EU per NM", 50)

    with column_3:
        plot_metric("Distance", total_distance, prefix="", suffix=" NM", show_graph=False)
        plot_gauge(total_eu_cost_pernm, "#FF2B2B", "euros", "EU Penalty", 90)
        
    with column_4:
        plot_metric("Time", total_time, prefix="", suffix=" Hours", show_graph=False)
        plot_gauge(total_speed, "#29B09D", " knots", "Average Speed", 31)

#GEN AI
# from pandasai import SmartDataframe
# from pandasai.llm import GooglePalm
# from pandasai.responses.response_parser import ResponseParser
# from pandasai.callbacks import BaseCallback
# class StreamlitCallback(BaseCallback):
#     def __init__(self, container) -> None:
#         """Initialize callback handler."""
#         self.container = container

#     def on_code(self, response: str):
#         self.container.code(response)

# class StreamlitResponse(ResponseParser):
#     def __init__(self, context) -> None:
#         super().__init__(context)

#     def format_dataframe(self, result):
#         st.dataframe(result["value"])
#         return

#     def format_plot(self, result):
#         st.image(result["value"])
#         return

#     def format_other(self, result):
#         st.write(result["value"])
#         return

# GOOGLE_API_KEY = st.secrets.API_KEY
# llm = GooglePalm(api_key=GOOGLE_API_KEY)

# st.write("# :compass:  Chat with Vessel voyage data")
# with st.expander(":badminton_racquet_and_shuttlecock: Dataframe preview"):
#     st.write(df1.tail(5))
# if df is not None:
#     sdf = SmartDataframe(df1,config={"llm":llm,"response_parser":StreamlitResponse})
#     # st.dataframe(df1.tail(3))
#     st.write("Some sample questions- Describe the data, dtypes of variables, shape of the data, any missing value, are there any duplicate rows, plot the graph, group the data by and calculate average ")
#     prompt = st.text_area("Enter your query")
#     if st.button("Generate"):
#         if prompt:
#             with st.spinner("Generating response..."):
#                 response = sdf.chat(prompt)
#                 st.success(response)

#                 st.set_option('deprecation.showPyplotGlobalUse', False)
#                 st.pyplot()
#         else:
#             st.warning("Please enter another query")
