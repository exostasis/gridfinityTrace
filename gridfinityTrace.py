import streamlit as st

app = st.navigation([
    st.Page("pages/welcome.py", title="Welcome"), 
    st.Page("pages/markerGeneration.py", title="ArUco Marker Generation"),
    st.Page("pages/svgTracing.py", title="Tracing")
])

app.run()