import streamlit as st
import pandas as pd


def render(P):
    st.title("Biometric Evaluation")
    st.info("Run the evaluation scripts on separate enrollment / validation / test sessions. The app deliberately does not invent accuracy numbers.")
    rows = P.db.fetch("SELECT modality, model, COUNT(*) AS attempts, AVG(score) AS mean_score FROM verification_results GROUP BY modality, model ORDER BY modality")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown("### Required final metrics")
    st.write(["FAR / FMR", "FRR / FNMR", "EER", "ROC-AUC", "confusion matrix", "95% confidence intervals", "latency", "quality-failure rate"])
