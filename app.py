# app.py
import streamlit as st
import requests

st.title("🛡️ Causal-Guard")

incident = st.text_area("Incident Description")

if st.button("Validate"):
    response = requests.post(
        "http://localhost:8000/validate",  # Change to Render URL when deployed
        json={"incident": incident}
    )
    result = response.json()
    
    st.write("**Explanation:**", result['explanation'])
    if result['admissible']:
        st.success("✅ Causally Admissible")
    else:
        st.error("❌ Causally Inadmissible")
        for v in result['violations']:
            st.write(f"- {v['constraint']}: {v['reason']}")