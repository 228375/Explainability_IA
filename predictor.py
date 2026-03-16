import joblib
import pandas as pd
import numpy as np

# Chargement des ressources pre-entrainees
model = joblib.load('rf_model.joblib')
le_source = joblib.load('le_source.joblib')
le_pos = joblib.load('le_pos.joblib')
explainer = joblib.load('shap_explainer.joblib')
df = pd.read_csv('HR_anonymized.csv')

def get_employee_prediction(emp_index):
    """
    Recupere les donnees d'un employe par son index et retourne un diagnostic complet.
    Inclut la probabilite de depart et les facteurs explicatifs SHAP.
    """
    # 1. Extraction des donnees brutes de l'employe
    try:
        emp_row = df.iloc[emp_index].copy()
    except IndexError:
        return {"erreur": "Index employe non valide"}

    # 2. Pre-traitement des variables calculees (Tenure)
    # On utilise la meme date de reference que lors de l'entrainement
    hire_date = pd.to_datetime(emp_row['DateofHire'])
    tenure = (pd.to_datetime('2020-01-01') - hire_date).days / 365.25

    # 3. Encodage des variables categorieles
    source_enc = le_source.transform([emp_row['RecruitmentSource']])[0]
    pos_enc = le_pos.transform([emp_row['Position']])[0]

    # 4. Preparation du vecteur d'entree pour le modele (11 colonnes)
    features_order = [
        'Salary', 'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount',
        'Absences', 'DaysLateLast30', 'DeptID', 'PerfScoreID',
        'Tenure', 'RecruitmentSource_Enc', 'Position_Enc'
    ]
    
    input_data = pd.DataFrame([{
        'Salary': emp_row['Salary'],
        'EngagementSurvey': emp_row['EngagementSurvey'],
        'EmpSatisfaction': emp_row['EmpSatisfaction'],
        'SpecialProjectsCount': emp_row['SpecialProjectsCount'],
        'Absences': emp_row['Absences'],
        'DaysLateLast30': emp_row['DaysLateLast30'],
        'DeptID': emp_row['DeptID'],
        'PerfScoreID': emp_row['PerfScoreID'],
        'Tenure': tenure,
        'RecruitmentSource_Enc': source_enc,
        'Position_Enc': pos_enc
    }])[features_order]

    # 5. Calcul de la probabilite
    prob = model.predict_proba(input_data)[0][1]

    # 6. Analyse SHAP pour l'explicabilite (pour l'Agent IA)
    shap_values = explainer(input_data)
    
    # Extraction de l'importance des features pour la classe "Depart"
    if len(shap_values.values.shape) == 3:
        vals = shap_values.values[0, :, 1]
    else:
        vals = shap_values.values[0]

    # Selection des deux facteurs les plus impactants
    feature_importance = dict(zip(features_order, vals))
    top_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
    
    # Nettoyage des noms de facteurs pour une meilleure lecture par le LLM
    clean_factors = [f"{f.replace('_Enc', '')}" for f, v in top_factors]

    # 7. Formatage du resultat pour le chatbot
    return {
        "statut": "Risque eleve" if prob > 0.5 else "Risque faible",
        "probabilite_depart": f"{round(prob * 100, 1)}%",
        "facteurs_cles": clean_factors,
        "details_techniques": {
            "poste": emp_row['Position'],
            "anciennete": f"{round(tenure, 1)} ans",
            "satisfaction": emp_row['EmpSatisfaction']
        }
    }