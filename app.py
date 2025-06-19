import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta

# --- 1. Model ve YENİ EKLENEN MAHALLE LİSTESİNİ Yükleme ---
try:
    artifacts = joblib.load('ultimate_noshow_model.joblib')
    model_pipeline = artifacts['model_pipeline']
    optimal_threshold = artifacts['optimal_threshold']
    # .get() metodu, eğer anahtar yoksa çökmesini engeller, varsayılan bir değer atar.
    neighbourhood_list = artifacts.get('neighbourhoods', ['JARDIM DA PENHA'])
except FileNotFoundError:
    st.error(
        "Model dosyası ('ultimate_noshow_model.joblib') bulunamadı. Lütfen önce `ultimate_train_model.py` betiğini çalıştırarak modeli eğitin.")
    st.stop()

# --- 2. Streamlit Arayüzü ---
st.set_page_config(page_title="Randevu Kaçak Tahmini", layout="centered", initial_sidebar_state="collapsed")
st.title("👨‍⚕️ Gelişmiş Randevu Kaçak Tahmin Modeli")
st.markdown(
    "Bu uygulama, hastanın temel bilgileri ve geçmiş randevu davranışlarını kullanarak, tıbbi randevusuna gelme olasılığını tahmin eder.")

# --- Kullanıcı Girdileri ---
st.header("Hasta ve Randevu Bilgileri")

st.write("Lütfen randevu tarihlerini giriniz:")
col_date1, col_date2 = st.columns(2)
with col_date1:
    scheduled_day = st.date_input("Randevunun Alındığı Gün", value=date.today())
with col_date2:
    appointment_day = st.date_input("Randevu Günü", value=date.today() + timedelta(days=7))

st.divider()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Cinsiyet", ["F", "M"], help="F: Kadın, M: Erkek")
    age = st.slider("Yaş", 0, 115, 35)
    hypertension = st.checkbox("Hipertansiyon")
    diabetes = st.checkbox("Diyabet")
    alcoholism = st.checkbox("Alkolizm")
    handicap = st.checkbox("Engellilik")
    sms_received = st.checkbox("SMS Hatırlatması Aldı")

with col2:
    # --- YENİ EKLENTİ: METİN KUTUSU YERİNE SEÇİM KUTUSU ---
    neighbourhood = st.selectbox("Mahalle", options=neighbourhood_list,
                                 index=neighbourhood_list.index('JARDIM DA PENHA'))

    prev_appointments = st.number_input("Hastanın Önceki Toplam Randevu Sayısı", min_value=0, value=0, step=1)
    prev_noshows = st.number_input("Hastanın Önceki Kaçırdığı Randevu Sayısı", min_value=0,
                                   max_value=prev_appointments if prev_appointments > 0 else 0, value=0, step=1)

# ... (Tahmin butonu ve geri kalan her şey aynı kalıyor) ...
if st.button("Riski Tahmin Et", type="primary", use_container_width=True):

    if scheduled_day > appointment_day:
        st.error("HATA: Randevunun alındığı tarih, randevu gününden sonra olamaz!")
    else:
        waiting_days = (appointment_day - scheduled_day).days

        input_data = {
            'Gender': 0 if gender == 'F' else 1,
            'Age': age,
            'Scholarship': 0,
            'Hypertension': int(hypertension),
            'Diabetes': int(diabetes),
            'Alcoholism': int(alcoholism),
            'Handicap': int(handicap),
            'SMS_Received': int(sms_received),
            'Neighbourhood': neighbourhood.upper(),
            'WaitingDays': waiting_days,
            'Weekday': pd.to_datetime(appointment_day).day_name(),
            'PrevNoShowCount': prev_noshows,
            'PrevAppointmentCount': prev_appointments,
            'NoShowRatio': (prev_noshows / (prev_appointments + 1e-6)),
            'HealthIssueCount': int(hypertension) + int(diabetes) + int(alcoholism)
        }
        X_predict = pd.DataFrame([input_data])

        if prev_appointments > 2 and prev_noshows == prev_appointments:
            st.subheader("Tahmin Sonucu (İş Kuralı Devrede)")
            st.error(f"⚠️ Yüksek Risk: Hasta, önceki {prev_appointments} randevusunun tamamını kaçırmıştır.")
            st.info("Bu tahmin, modelin kararı yerine doğrudan tanımlanan iş kuralına göre yapılmıştır.")
        else:
            probability_noshow = model_pipeline.predict_proba(X_predict)[:, 1][0]
            prediction = 1 if probability_noshow >= optimal_threshold else 0

            st.subheader("Tahmin Sonucu (Model Tahmini)")
            if prediction == 1:
                st.error(f"⚠️ Yüksek Risk: Hastanın randevuya **GELMEME** olasılığı %{probability_noshow * 100:.2f}")
            else:
                st.success(
                    f"✅ Düşük Risk: Hastanın randevuya **GELME** olasılığı %{(1 - probability_noshow) * 100:.2f}")

            st.info(
                f"Bu tahmin, F1-Skorunu maksimize eden `{optimal_threshold:.2f}` olasılık eşiğine göre yapılmıştır.")