import mlflow
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json

# [Bagian Load Model & Kolom Tetap Sama seperti Kodemu]
encoder_Credit_Mix = joblib.load("model/encoder_Credit_Mix.joblib")
encoder_Payment_Behaviour = joblib.load("model/encoder_Payment_Behaviour.joblib")
encoder_Payment_of_Min_Amount = joblib.load("model/encoder_Payment_of_Min_Amount.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")

scaler_Age = joblib.load("model/scaler_Age.joblib")
scaler_Num_Bank_Accounts = joblib.load("model/scaler_Num_Bank_Accounts.joblib")
scaler_Num_Credit_Card = joblib.load("model/scaler_Num_Credit_Card.joblib")
scaler_Interest_Rate = joblib.load("model/scaler_Interest_Rate.joblib")
scaler_Num_of_Loan = joblib.load("model/scaler_Num_of_Loan.joblib")
scaler_Delay_from_due_date = joblib.load("model/scaler_Delay_from_due_date.joblib")
scaler_Num_of_Delayed_Payment = joblib.load("model/scaler_Num_of_Delayed_Payment.joblib")
scaler_Changed_Credit_Limit = joblib.load("model/scaler_Changed_Credit_Limit.joblib")
scaler_Num_Credit_Inquiries = joblib.load("model/scaler_Num_Credit_Inquiries.joblib")
scaler_Outstanding_Debt = joblib.load("model/scaler_Outstanding_Debt.joblib")
scaler_Credit_History_Age = joblib.load("model/scaler_Credit_History_Age.joblib")

scaler_Monthly_Inhand_Salary = joblib.load("model/scaler_Monthly_Inhand_Salary.joblib")
scaler_Monthly_Balance = joblib.load("model/scaler_Monthly_Balance.joblib")
scaler_Amount_invested_monthly = joblib.load("model/scaler_Amount_invested_monthly.joblib")
scaler_Total_EMI_per_month = joblib.load("model/scaler_Total_EMI_per_month.joblib")

pca_numerical_columns_1 = [
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
    'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_History_Age'
]
 
pca_numerical_columns_2 = [
    "Monthly_Inhand_Salary", "Monthly_Balance", "Amount_invested_monthly", "Total_EMI_per_month"
]
 
def data_preprocessing(data):
    data = data.copy()
    df = pd.DataFrame()
    
    # 1. Transformasi Age
    df["Age"] = scaler_Age.transform(data[["Age"]]).flatten()
    
    # 2. Transformasi Categorical (Menggunakan kurung siku ganda [[]] agar tetap 2D)
    df["Credit_Mix"] = encoder_Credit_Mix.transform(data[["Credit_Mix"]]).flatten()
    df["Payment_of_Min_Amount"] = encoder_Payment_of_Min_Amount.transform(data[["Payment_of_Min_Amount"]]).flatten()
    df["Payment_Behaviour"] = encoder_Payment_Behaviour.transform(data[["Payment_Behaviour"]]).flatten()
    
    # 3. PCA 1: Skalakan fitur satu per satu langsung ke dataframe `data` asli
    data["Num_Bank_Accounts"] = scaler_Num_Bank_Accounts.transform(data[["Num_Bank_Accounts"]]).flatten()
    data["Num_Credit_Card"] = scaler_Num_Credit_Card.transform(data[["Num_Credit_Card"]]).flatten()
    data["Interest_Rate"] = scaler_Interest_Rate.transform(data[["Interest_Rate"]]).flatten()
    data["Num_of_Loan"] = scaler_Num_of_Loan.transform(data[["Num_of_Loan"]]).flatten()
    data["Delay_from_due_date"] = scaler_Delay_from_due_date.transform(data[["Delay_from_due_date"]]).flatten()
    data["Num_of_Delayed_Payment"] = scaler_Num_of_Delayed_Payment.transform(data[["Num_of_Delayed_Payment"]]).flatten()
    data["Changed_Credit_Limit"] = scaler_Changed_Credit_Limit.transform(data[["Changed_Credit_Limit"]]).flatten()
    data["Num_Credit_Inquiries"] = scaler_Num_Credit_Inquiries.transform(data[["Num_Credit_Inquiries"]]).flatten()
    data["Outstanding_Debt"] = scaler_Outstanding_Debt.transform(data[["Outstanding_Debt"]]).flatten()
    data["Credit_History_Age"] = scaler_Credit_History_Age.transform(data[["Credit_History_Age"]]).flatten()
    
    # Transformasikan ke PCA 1
    df[["pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5"]] = pca_1.transform(data[pca_numerical_columns_1])
    
    # 4. PCA 2: Skalakan fitur kelompok kedua
    data["Monthly_Inhand_Salary"] = scaler_Monthly_Inhand_Salary.transform(data[["Monthly_Inhand_Salary"]]).flatten()
    data["Monthly_Balance"] = scaler_Monthly_Balance.transform(data[["Monthly_Balance"]]).flatten()
    data["Amount_invested_monthly"] = scaler_Amount_invested_monthly.transform(data[["Amount_invested_monthly"]]).flatten()
    data["Total_EMI_per_month"] = scaler_Total_EMI_per_month.transform(data[["Total_EMI_per_month"]]).flatten()
    
    # Transformasikan ke PCA 2
    df[["pc2_1", "pc2_2"]] = pca_2.transform(data[pca_numerical_columns_2])
    
    return df

def prediction(data):
    # 1. Jika data berupa DataFrame, bungkus ke format split dan ubah jadi STRING JSON
    if isinstance(data, pd.DataFrame):
        payload = {
            "dataframe_split": {
                "columns": data.columns.tolist(),
                "data": data.values.tolist()
            }
        }
        json_payload = json.dumps(payload) # Menjadi teks biasa, aman dari pengecekan 'or {}'
    else:
        # Jika input sudah berupa dictionary, ubah ke string JSON
        json_payload = json.dumps(data) if isinstance(data, dict) else data

    # 2. Sesuaikan PORT dengan MLflow serve yang aktif (misal 5001 atau 5005)
    url = "http://localhost:5004/invocations" 
    headers = {"Content-Type": "application/json"}
    
    # 3. Kirim json_payload yang sudah berwujud string murni
    response = requests.post(url, data=json_payload, headers=headers)
    
    if response.status_code == 200:
        res_json = response.json()
        predictions = res_json.get("predictions")
        
        result_target = joblib.load("model/encoder_target.joblib")
        final_result = result_target.inverse_transform(predictions)
        return final_result
    else:
        return f"Error Server MLflow ({response.status_code}): {response.text}"

# [TARUH DI BAGIAN PALING BAWAH FILE PREPROCESSAPI.PY]

if __name__ == "__main__":
    # 1. Buat contoh data mentah (sesuaikan nama kolomnya dengan dataset aslimu)
    data_mentah = pd.DataFrame([{
        "Age": 23,
        "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "High_spent_Small_value_payments",
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 4,
        "Interest_Rate": 9,
        "Num_of_Loan": 4,
        "Delay_from_due_date": 5,
        "Num_of_Delayed_Payment": 7,
        "Changed_Credit_Limit": 11.27,
        "Num_Credit_Inquiries": 4,
        "Outstanding_Debt": 800.0,
        "Credit_History_Age": 220,
        "Monthly_Inhand_Salary": 3000.0,
        "Monthly_Balance": 310.0,
        "Amount_invested_monthly": 120.0,
        "Total_EMI_per_month": 49.5
    }])

    print("--- Memulai Preprocessing Data Mentah ---")
    # LANGKAH KRUSIAL: Ubah dulu data mentah menjadi komponen PCA & Skala Scaler
    data_siap_prediksi = data_preprocessing(data_mentah)
    
    print("\n--- Struktur Data Setelah Preprocessing (Siap Kirim): ---")
    print(data_siap_prediksi)

    print("\n--- Mengirim Data ke Server MLflow ---")
    # Kirim data yang SUDAH di-preprocessing, bukan data_mentah
    hasil_prediksi = prediction(data_siap_prediksi)
    
    print("\n--- Hasil Prediksi Akhir: ---")
    print(hasil_prediksi)