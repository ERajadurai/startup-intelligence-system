from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Trained Brain
with open('startup_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    le_ind, le_cou, le_sta = data['le_ind'], data['le_cou'], data['le_sta']

@app.route('/')
def home():
    return render_template('index.html', industries=le_ind.classes_, countries=le_cou.classes_, stages=le_sta.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            le_ind.transform([request.form['industry']])[0],
            le_cou.transform([request.form['country']])[0],
            le_sta.transform([request.form['stage']])[0],
            float(request.form['funding']),
            float(request.form['employees']),
            float(request.form['revenue']),
            float(request.form['growth'])
        ]
        prediction = model.predict([features])
        prob = model.predict_proba([features])[0][1] * 100
        res = "Likely Success" if prediction[0] == 1 else "High Risk"
        return render_template('result.html', prediction_text=res, confidence=f"{prob:.2f}%")
    except Exception as e:
        return f"Backend Error: {e}"

@app.route('/market')
def market():
    mg_df = pd.read_csv('market_gap.csv')
    mg_df['Score'] = (mg_df['MarketSize'] * mg_df['GrowthRate']) / (mg_df['CompetitionIntensity'] + 1)
    gaps = mg_df.sort_values(by='Score', ascending=False).head(10).to_dict('records')
    return render_template('market.html', gaps=gaps)

if __name__ == "__main__":
    app.run(debug=True)