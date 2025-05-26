# Student Option Classification API

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

This project provides a RESTful API for predicting the academic track (Literary, Scientific, or Organization/Society/Economy) for high school students in Fianarantsoa, Madagascar. The system uses machine learning (PyCaret) to make predictions based on academic performance and other relevant factors.

## 🚀 Key Features

- **Accurate Predictions**: Machine learning model trained on historical student data
- **Easy Integration**: Simple REST API endpoints for easy integration with other systems
- **Detailed Insights**: Returns prediction probabilities for each possible track
- **Scalable**: Built with FastAPI for high performance
- **Comprehensive Logging**: Built-in logging for monitoring and debugging

## 📋 Table of Contents
- [🚀 Key Features](#-key-features)
- [📁 Project Structure](#-project-structure)
- [🛠 Prerequisites](#-prerequisites)
- [🚀 Installation](#-installation)
- [🚦 Usage](#-usage)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Training the Model](#training-the-model)
  - [Running the API](#running-the-api)
- [🌐 API Endpoints](#-api-endpoints)
  - [Health Check](#health-check)
  - [Make a Prediction](#make-a-prediction)
  - [Get Model Features](#get-model-features)
- [🧪 Testing](#-testing)
- [⚙️ Configuration](#️-configuration)
- [🐳 Deployment](#-deployment)
- [🤝 Contributing](#-contributing)

## ✨ Features

- **Machine Learning Model**: Predicts student academic tracks using PyCaret
- **RESTful API**: Easy-to-use endpoints for predictions
- **Scalable Architecture**: Modular design for easy maintenance and extension
- **Comprehensive Documentation**: Includes API documentation and examples
- **Health Monitoring**: Built-in health check endpoint
- **Input Validation**: Robust validation of input data
- **Logging**: Comprehensive logging for debugging and monitoring

## 📁 Project Structure

```
model_deploiment_using_apiRest/
├── app/                               # Application source code
│   ├── api/                           # API endpoints and routes
│   │   └── endpoints.py               # API route handlers
│   ├── core/                          # Core application configuration
│   │   └── config.py                  # Application settings and configuration
│   ├── models/                        # Data models and schemas
│   │   └── schemas.py                 # Pydantic models for request/response validation
│   ├── services/                      # Business logic and services
│   │   ├── __init__.py
│   │   └── prediction_service.py      # Prediction service implementation
│   └── main.py                        # FastAPI application instance
│
├── data/                              # Data files
│   └── raw/                           # Raw data files
│       └── base.xlsx                  # Input dataset
│
├── models/                            # Trained models
│   └── option_classifier.pkl          # Serialized model file
│
├── tests/                             # Test files
│   ├── __pycache__/
│   └── test_api.py                    # API test cases
│
├── .coverage                          # Test coverage report
├── .dockerignore                      # Docker ignore file
├── .env                               # Environment variables
├── .git/                              # Git repository
├── .gitignore                         # Git ignore file
├── .pytest_cache/                     # Pytest cache
├── Dockerfile                         # Docker configuration
├── README.md                          # Project documentation
├── api_look.png                       # API documentation screenshot
├── docker-compose.yml                 # Docker Compose configuration
├── inspect_schema.py                  # Utility for inspecting data schema
├── logs.log                           # Application logs
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Python dependencies
├── run.py                             # Application entry point
├── setup.py                           # Package configuration
├── test_endpoint.py                   # Endpoint testing script
├── test_model.py                      # Model testing script
├── train_model.py                     # Model training script
└── training.log                       # Training logs
```

## 🛠 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)
- Git (for version control)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/model_deploiment_using_apiRest.git
   cd model_deploiment_using_apiRest
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Copy the example `.env` file and update the values as needed:
   ```bash
   cp .env.example .env
   ```

## 🚦 Usage

### Setting Up the Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/optionclass-api.git
   cd optionclass-api
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

### Training the Model

To train a new model:

```bash
python train_model.py
```

This will:
1. Load and preprocess the data
2. Train a new model using PyCaret
3. Save the model to the specified path in `.env`
4. Generate training logs in `training.log`

### Running the API

To start the API server in development mode:

```bash
uvicorn app.main:app --reload
```

For production:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` by default.

### API Documentation

Once the API is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🌐 API Endpoints

### Health Check

```
GET /api/v1/health
```

Check if the API is running and if the model is loaded.

**Example Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "model_features": ["sexe", "nb_frère", "nb_sœur", "commune_d_origine", ...]
}
```

### Make a Prediction

```
POST /api/v1/predict
```

**Request Body:**
```json
{
  "sexe": 1.0,
  "nb_frère": 2.0,
  "nb_sœur": 1.0,
  "commune_d_origine": 25,
  "habite_avec_les_parents": 1,
  "electricite": 1,
  "conn_sur_les_options": 1,
  "MLG": 15.0,
  "FRS": 14.5,
  "ANG": 16.0,
  "HG": 13.5,
  "SES": 12.0,
  "MATHS": 18.0,
  "PC": 17.5,
  "SVT": 16.5,
  "EPS": 15.0,
  "première_S": 14.5,
  "deuxième_S": 15.5,
  "MOY_AN": 15.0
}
```

**Successful Response (200 OK):**
```json
{
  "prediction": "S",
  "confidence_scores": {
    "L": 0.1,
    "S": 0.8,
    "OSE": 0.1
  },
  "status": "success"
}
```

**Error Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "loc": ["body", "sexe"],
      "msg": "ensure this value is less than or equal to 1",
      "type": "value_error.number.not_le",
      "ctx": {"limit_value": 1}
    }
  ]
}
```

### Get Model Features

```
GET /api/v1/model/features
```

Get the list of features expected by the model.

**Example Response:**
```json
{
  "features": ["sexe", "nb_frère", "nb_sœur", "commune_d_origine", ...],
  "required": true,
  "status": "success"
}
```

## 🧪 Testing

To run the test suite:

```bash
pytest tests/ -v --cov=app
```

This will run all tests and generate a coverage report.

## ⚙️ Configuration

Configuration is managed through environment variables in the `.env` file:

```ini
# Application settings
APP_NAME="OptionClass API"
APP_VERSION="1.0.0"
DEBUG=True

# Server settings
HOST="0.0.0.0"
PORT=8000

# Model settings
MODEL_PATH="models/option_classifier.pkl"

# Logging
LOG_LEVEL="INFO"
LOG_FILE="app.log"

# Data settings
DATA_PATH="data/raw/base.xlsx"
PROCESSED_DATA_PATH="data/processed/processed_data.csv"
```

## 🧪 Testing

To run tests:

```bash
pytest tests/
```

## 🚀 Deployment

### Production

For production deployment, consider using:

1. **Gunicorn with Uvicorn Workers**
   ```bash
   pip install gunicorn
   gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 app.main:app
   ```

2. **Docker**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]
   ```

3. **Cloud Platforms**
   - AWS Elastic Beanstalk
   - Google Cloud Run
   - Microsoft Azure App Service
   - Heroku

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

For questions or support, please contact [Your Name] at [your.email@example.com].

---

![API Interface](/api_look.png)
