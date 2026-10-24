# RAGDiagnose Streamlit Demo

## Deploy to Streamlit Community Cloud
1. Upload `streamlit_app.py` and `requirements.txt` to the root of a GitHub repository.
2. Open https://share.streamlit.io/ and create an app from that repository.
3. Set Main file path to `streamlit_app.py`.
4. In app Settings → Secrets, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy the app, open Knowledge Base, upload a text-based PDF, process it, and ask a question in RAG Chat.

This is a self-contained demo and does not require the separate Render backend.
It uses lightweight lexical retrieval, so the scores and root-cause diagnoses are heuristic demo signals rather than formal RAG evaluation metrics.
