import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests

# Set judul halaman web
st.title("Aplikasi Web Credit Scoring")
st.write("Masukkan data nasabah di bawah ini untuk melihat hasil prediksi.")

# --- Bagian Input Form di Browser ---
credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Poor"])
payment_min = st.selectbox("Payment of Min Amount", ["No", "Yes"])
payment_behaviour = st.selectbox("Payment Behaviour", [
    "Low_spent_Small_value_payments", 
    "High_spent_Medium_value_payments", 
    "Low_spent_Medium_value_payments",
    "High_spent_Large_value_payments"
])
age = st.number_input("Age", min_value=17, max_value=100, value=23)
num_bank = st.number_input("Number of Bank Accounts", min_value=0, value=3)
num_card = st.number_input("Number of Credit Cards", min_value=0, value=4)
interest_rate = st.number_input("Interest Rate", min_value=0.0, value=3.0)
num_loan = st.number_input("Number of Loans", min_value=0, value=4)
delay_due = st.number_input("Delay from Due Date", min_value=0, value=3)
num_delayed_pay = st.number_input("Number of Delayed Payments", min_value=0, value=7)
changed_limit = st.number_input("Changed Credit Limit", min_value=0.0, value=11.27)
num_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=5)
outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=809.98)
monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=1824.80)
monthly_balance = st.number_input("Monthly Balance", min_value=0.0, value=186.26)
amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=236.64)
total_emi = st.number_input("Total EMI per Month", min_value=0.0, value=49.50)
credit_age = st.number_input("Credit History Age (Months)", min_value=0, value=216)

# --- Tombol untuk Prediksi ---
if st.button("Cek Credit Scoring"):
    
    # 1. Bungkus inputan menjadi DataFrame
    columns = [
        "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Age", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Monthly_Inhand_Salary", "Monthly_Balance",
        "Amount_invested_monthly", "Total_EMI_per_month", "Credit_History_Age"
    ]
    raw_data = [credit_mix, payment_min, payment_behaviour, age, num_bank, num_card, interest_rate, num_loan,
                delay_due, num_delayed_pay, changed_limit, num_inquiries, outstanding_debt, monthly_salary,
                monthly_balance, amount_invested, total_emi, credit_age]
    
    df = pd.DataFrame([raw_data], columns=columns)
    
    # 2. Panggil fungsi preprocessing (Load transformer lokal kamu)
    # *Catatan: Pastikan script ini bisa membaca file joblib kamu*
    try:
        # Load object joblib (Disederhanakan, pastikan jalurnya benar)
        scaler_Age = joblib.load("model/scaler_Age.joblib")
        encoder_Credit_Mix = joblib.load("model/encoder_Credit_Mix.joblib")
        encoder_Payment_of_Min_Amount = joblib.load("model/encoder_Payment_of_Min_Amount.joblib")
        encoder_Payment_Behaviour = joblib.load("model/encoder_Payment_Behaviour.joblib")
        
        # Simulasi transform sederhana untuk demo interface
        df_preprocessed = pd.DataFrame()
        df_preprocessed["Age"] = scaler_Age.transform(np.asarray(df["Age"]).reshape(-1, 1)).flatten()
        df_preprocessed["Credit_Mix"] = encoder_Credit_Mix.transform(df["Credit_Mix"])
        df_preprocessed["Payment_of_Min_Amount"] = encoder_Payment_of_Min_Amount.transform(df["Payment_of_Min_Amount"])
        df_preprocessed["Payment_Behaviour"] = encoder_Payment_Behaviour.transform(df["Payment_Behaviour"])
        
        # ... (Tambahkan sisa transformasi PCA & Scaler milikmu di sini agar sinkron) ...
        # Untuk keperluan contoh, kita asumsikan output df_preprocessed sudah siap dikirim
        
        # 3. Kirim ke MLflow Server (Port 5001)
        url = "http://localhost:5001/invocations"
        json_output = {
            "dataframe_split": {
                "columns": df_preprocessed.columns.tolist(),
                "data": df_preprocessed.values.tolist()
            }
        }
        
        response = requests.post(url, json=json_output)
        
        if response.status_code == 200:
            res_json = response.json()
            predictions = res_json.get("predictions")
            
            # Inverse transform target hasil prediksi
            result_target = joblib.load("model/encoder_target.joblib")
            final_result = result_target.inverse_transform(predictions)
            
            # Tampilkan hasil di browser!
            st.success(f"Hasil Prediksi Kelayakan Kredit: **{final_result[0]}**")
        else:
            st.error(f"Gagal terhubung ke MLflow Server. Status: {response.status_code}")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")