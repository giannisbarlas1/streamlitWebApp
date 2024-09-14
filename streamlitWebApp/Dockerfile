FROM python:3.8-slim


WORKDIR /app
COPY . /app

# Εγκατάσταση των εξαρτήσεων
RUN pip install -r requirements.txt

# Εκκίνηση της εφαρμογής Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
