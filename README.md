# Clinical Trial Success Predictor

A web application that predicts whether a clinical trial will be completed based on study characteristics, with an integrated AI chatbot assistant.

## Features

- **Interactive Prediction Form**: Input trial parameters and get instant predictions
- **Real-time Analysis**: XGBoost model with TF-IDF text analysis
- **Visual Results**: Probability scores, key factors, and color-coded outcomes
- **AI Chatbot**: Ask questions about predictions, get recommendations, and learn about trial success factors
- **Actionable Recommendations**: Personalized suggestions to improve trial success probability

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Prepare model files**:
Place these files in the same directory as `app.py`:
- `xgb_tfidf_model.pkl`
- `onehot_encoder.pkl`
- `scaler.pkl`
- `tfidf_vectorizer.pkl`
- `sponsor_map.pkl`

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Access the app**:
Open your browser to `http://localhost:8501`

## Usage

### Making Predictions

1. Fill in the trial information form:
   - Brief Summary (include any known challenges)
   - Conditions and Interventions
   - Study design parameters
   - Enrollment and duration

2. Click "Predict Trial Outcome"

3. View results:
   - Completion probability
   - Key risk/success factors
   - Personalized recommendations

### Using the Chatbot

Ask the AI assistant about:
- "Explain the prediction"
- "What are the key factors?"
- "Give me recommendations"
- "How can I improve enrollment?"
- "What about trial phases?"
- "Tell me about sponsor impact"

## Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: 
  - Text analysis (TF-IDF on summaries, conditions, interventions)
  - Categorical features (phase, study type, design, demographics)
  - Numeric features (enrollment, duration, risk scores)
  - Sponsor success rates
- **Decision Threshold**: 0.6
- **Training Data**: Industry-funded clinical trials

## Tips for Best Results

- Be detailed in the Brief Summary
- Mention any recruitment challenges, funding issues, or operational concerns
- Provide accurate enrollment and timeline estimates
- Use the chatbot to understand how different factors affect predictions
