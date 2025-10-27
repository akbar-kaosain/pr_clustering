import os
import streamlit as st

st.set_page_config(page_title="PR Clustering (smoke test)")

st.title("Hello from Streamlit 👋")
st.write("If you can see this, Streamlit is rendering correctly.")

# Secrets / token check
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
if GITHUB_TOKEN:
    st.success("✅ GitHub token loaded.")
else:
    st.info("ℹ️ No GitHub token found. Add it to .streamlit/secrets.toml or env var.")

# Make sure some UI always renders:
st.text_input("Type anything here:")
st.button("Click me")
