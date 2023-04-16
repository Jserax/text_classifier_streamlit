bentoml build -f ./bentofile.yaml
bentoml serve text_classifier:latest --host localhost --port
streamlit app.py --server.port 8000
