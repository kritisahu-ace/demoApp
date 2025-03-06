import streamlit as st
from sales_country import country_analysis
from sales_state_wise import state_analysis
from sales_district_wise import district_analysis
from price_country import price_country_analysis
from price_state_wise import price_state_analysis
from price_district_wise import price_district_analysis
from sales_type_country import sales_type_country_analysis
from sales_type_state_wise import sales_type_state_analysis
from sales_type_district_wise import sales_type_district_analysis
from change_car_country import customer_country_analysis

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("Used Car Sales Forecasting")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Top Selling Car Model", "Preferred Price Range", "Preferred Sales Type", "Frequent Visiting Customer"])

with tab1:
    #st.header("Top Selling Car Model (Next Quarter)")
    
    # Sub-tabs for Top Selling Car Model
    subtab1, subtab2, subtab3 = st.tabs(["Country", "State-wise", "District-wise"])
    
    with subtab1:
        st.header("Top Selling Car Models by Country")
        plt = country_analysis()
        # st.plotly_chart(plt.gcf())  # Display the graph
        st.altair_chart(plt)
    
    with subtab2:
        st.header("Top Selling Car Models by State")
        plt = state_analysis()
        st.altair_chart(plt)

        # st.pyplot(plt.gcf())  # Display the graph
    
    with subtab3:
        st.header("Top Selling Car Models by District")
        plt = district_analysis()
        # st.pyplot(plt.gcf())  # Display the graph
        st.altair_chart(plt)


with tab2:
    #st.header("Preferred Price Range (Next Quarter)")
    
    # Sub-tabs for Preferred Price Range
    subtab4, subtab5, subtab6 = st.tabs(["Country", "State-wise", "District-wise"])
    
    with subtab4:
        st.header("Preferred Price Ranges by Country")
        plt = price_country_analysis()
        # st.pyplot(plt.gcf())  # Display the graph
        st.altair_chart(plt)

    
    with subtab5:
        st.header("Preferred Price Ranges by State")
        plt = price_state_analysis()
        # st.pyplot(plt.gcf())  # Display the graph
        st.altair_chart(plt)
    
    with subtab6:
        st.header("Preferred Price Ranges by District")
        plt = price_district_analysis()
        # st.pyplot(plt.gcf())  # Display the graph
        st.altair_chart(plt)


with tab3:
    #st.header("Preferred Sales Type (Next Quarter)")
    
    # Sub-tabs for Preferred Price Range
    subtab7, subtab8, subtab9 = st.tabs(["Country", "State-wise", "District-wise"])
    
    with subtab7:
        st.header("Preferred Sales Type by Country")
        plt = sales_type_country_analysis()
        st.altair_chart(plt)
        # st.pyplot(plt.gcf())  # Display the graph
    
    with subtab8:
        st.header("Preferred Sales Type by State")
        plt = sales_type_state_analysis()
        st.altair_chart(plt)
        # st.pyplot(plt.gcf())  # Display the graph
    
    with subtab9:
        st.header("Preferred Sales Type by District")
        plt = sales_type_district_analysis()
        st.altair_chart(plt)
        # st.pyplot(plt.gcf())  # Display the graph

with tab4:
    #st.header("Preferred Sales Type (Next Quarter)")
    
    # Sub-tabs for Preferred Price Range
    subtab10 = st.tabs(["Country"])
    
    with subtab10:
        st.header("Frequent visting customer by Country")
        plt = customer_country_analysis()
        # st.pyplot(plt.gcf())  # Display the graph
        st.altair_chart(plt)

    
    