import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Data load (Ensure unicorns.csv is in the same folder)
try:
    df = pd.read_csv('unicorns.csv')
    
    # Success label logic
    df['SuccessStatus'] = (df['Valuation_B'] > 2.0).astype(int)

    # Encoding
    le_ind, le_cou, le_sta = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['Industry'] = le_ind.fit_transform(df['Industry'])
    df['Country'] = le_cou.fit_transform(df['Country'])
    df['FundingStage'] = le_sta.fit_transform(df['FundingStage'])

    # Train
    X = df[['Industry', 'Country', 'FundingStage', 'TotalFunding_M', 'EmployeeCount', 'Revenue_M', 'GrowthRate']]
    y = df['SuccessStatus']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save
    with open('startup_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_ind': le_ind, 'le_cou': le_cou, 'le_sta': le_sta}, f)
    
    print("✅ Success: model_train.py worked! Brain ready.")
except Exception as e:
    print(f"❌ Error: {e}")