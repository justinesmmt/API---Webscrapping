import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate(r'C:\Users\justi\OneDrive - Fondation EPF\Documents\5A semestre 1 EPF\Data sources\repo\api-webscrapping-e7967-firebase-adminsdk-tiigz-02eaac3e18.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
parameters_ref = db.collection('parameters').document('parameters')

data = {
    'n_estimators': 100,
    'criterion': 'gini'
}
parameters_ref.set(data)