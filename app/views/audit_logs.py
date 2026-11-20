import streamlit as st
from app.common import df


def render(P):
    st.title("Audit Logs")
    kind = st.selectbox("Filter", ["all", "auth_attempt", "enroll", "samples_added", "user_deleted"])
    rows = P.audit.recent(500, None if kind == "all" else f"%{kind}%")
    st.dataframe(df(rows), use_container_width=True)
