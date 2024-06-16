#Answer for qn6 is here.
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install pipenv
RUN pip install pipenv

# Install dependencies
RUN pipenv install --deploy --system

COPY 'starter.py' .

CMD ["python3", "starter.py","2023","05"]

#Commands used to run this docker script
#docker build -t my_model_loader_image .
#docker run --rm my_model_loader_image