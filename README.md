# HR AI Explainer 🤖

An AI-powered employee turnover prediction system with explainable AI (SHAP) and LLM-generated explanations.

## Features

- **ML Model**: Random Forest classifier for predicting employee turnover
- **Explainability**: SHAP values to understand why an employee might leave
- **LLM Integration**: Uses Ollama (Gemma 3) for natural language explanations
- **Web Interface**: Modern frontend to explore predictions

## Project Structure

```
rh_ai_explainer/
├── app/
│   ├── main.py              # FastAPI application
│   ├── predictor.py         # ML prediction logic
│   ├── ollama_client.py    # LLM integration
│   ├── prompts.py           # Prompt templates
│   ├── schemas.py          # Pydantic schemas
│   ├── settings.py         # Configuration
│   └── templates/
│       └── index.html      # Frontend UI
├── HR_anonymized.csv       # HR Dataset
├── rf_model.joblib         # Trained Random Forest model
├── shap_explainer.joblib  # SHAP explainer
├── le_source.joblib       # Label encoder for recruitment sources
├── le_pos.joblib          # Label encoder for positions
└── requirements.txt       # Python dependencies
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rh_ai_explainer
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install Ollama** (for LLM explanations):
   - Download from: https://ollama.com
   - Run: `ollama pull gemma3:12b`

## Running the Application

### Start the API server:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Access the web interface:
- Open browser: http://127.0.0.1:8000/

### API Documentation:
- Swagger UI: http://127.0.0.1:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend |
| `/health` | GET | Health check |
| `/employees/count` | GET | Total employee count |
| `/predict/{index}` | GET | Get prediction + SHAP factors |
| `/predict/{index}/explain` | GET | Get prediction with LLM explanation |

## Example Usage

### Get prediction for employee:
```bash
curl http://127.0.0.1:8000/predict/0
```

### Get full explanation:
```bash
curl http://127.0.0.1:8000/predict/0/explain
```

## Configuration

Edit `app/settings.py` to customize:
- Ollama model
- Temperature
- API settings

## Technology Stack

- **Backend**: FastAPI (Python)
- **ML**: scikit-learn, SHAP
- **LLM**: Ollama (Gemma 3)
- **Frontend**: HTML/CSS/JavaScript

## License

MIT
