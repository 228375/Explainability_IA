import joblib
import pandas as pd
import numpy as np

# On charge les outils que tu as sauvegardés
model = joblib.load('rf_model.joblib')
explainer = joblib.load('shap_explainer.joblib')
df = pd.read_csv('HR_anonymized.csv')

def get_employee_prediction(emp_index):
    """
    Fonction que le chatbot va appeler.
    Prend l'index d'un employé, renvoie un diagnostic complet.
    """
    # 1. Extraire les données de l'employé (les mêmes features que l'entraînement)
    features = ['Salary', 'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 
                'Absences', 'DaysLateLast30', 'DeptID', 'PerfScoreID']
    
    employee_data = df.iloc[[emp_index]][features]
    
    # 2. Prédiction de probabilité (0 à 1)
    # [0][1] car on veut la probabilité de la classe 1 (départ)
    prob = model.predict_proba(employee_data)[0][1]
    
    # 3. Explication SHAP
    # On calcule l'impact de chaque critère pour CET employé
    shap_values = explainer(employee_data)
    
    # On isole les valeurs pour la classe 1
    if len(shap_values.values.shape) == 3:
        vals = shap_values.values[0, :, 1]
    else:
        vals = shap_values.values[0]

    feature_importance = dict(zip(features, vals))
    
    # On trie pour avoir les 2 facteurs les plus influents
    top_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:2]

    # 4. Résultat propre pour le Chatbot (format JSON/Dictionnaire)
    return {
        "statut": "Risque élevé" if prob > 0.5 else "Risque faible",
        "probabilite_depart": f"{round(prob * 100, 1)}%",
        "facteurs_cles": [f"{f} (impact: {round(v, 2)})" for f, v in top_factors],
        "conseil": "Une discussion avec le manager est recommandée." if prob > 0.5 else "Maintenir le suivi actuel."
    }