# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

print("Nihai model eğitim süreci başlatıldı...")

# --- 1. Veri Yükleme ve Temel Temizlik ---
try:
    df = pd.read_csv("KaggleV2-May-2016.csv")
except FileNotFoundError:
    print("HATA: 'KaggleV2-May-2016.csv' dosyası bulunamadı. Lütfen dosyanın doğru klasörde olduğundan emin olun.")
    exit()

# --- YENİ EKLENTİ: MAHALLE LİSTESİNİ ALMA ---
# Kullanıcı arayüzünde göstermek için tüm benzersiz ve sıralanmış mahallelerin listesini al.
neighbourhood_list = sorted(df['Neighbourhood'].unique().tolist())
print(f"{len(neighbourhood_list)} adet benzersiz mahalle bulundu ve kaydedilmeye hazırlandı.")

df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMS_Received',
                   'No-show': 'No_show'}, inplace=True)
df['No_show'] = df['No_show'].map({'Yes': 1, 'No': 0})
df = df[df['Age'].between(0, 115)].copy()

# ... (Özellik Mühendisliği ve diğer tüm adımlar aynı kalıyor) ...
for c in ['ScheduledDay', 'AppointmentDay']:
    df[c] = pd.to_datetime(df[c])
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
df['Weekday'] = df['AppointmentDay'].dt.day_name()

df_sorted = df.sort_values(['PatientId', 'AppointmentDay'])
group = df_sorted.groupby('PatientId')['No_show']
df_sorted['PrevNoShowCount'] = group.shift().fillna(0).cumsum()
df_sorted['PrevAppointmentCount'] = df_sorted.groupby('PatientId').cumcount()
df_sorted['NoShowRatio'] = (df_sorted['PrevNoShowCount'] / (df_sorted['PrevAppointmentCount'] + 1e-6)).fillna(0)
df = df.merge(df_sorted[['AppointmentID', 'PrevNoShowCount', 'PrevAppointmentCount', 'NoShowRatio']],
              on='AppointmentID', how='left')

df['HealthIssueCount'] = df[['Hypertension', 'Diabetes', 'Alcoholism']].sum(axis=1)
df['Handicap'] = (df['Handicap'] > 0).astype(int)
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

df.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], inplace=True)

categorical_features = ['Neighbourhood', 'Weekday']
numerical_features = ['Age', 'WaitingDays', 'HealthIssueCount', 'PrevNoShowCount', 'PrevAppointmentCount',
                      'NoShowRatio']

X = df.drop('No_show', axis=1)
y = df['No_show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Veri, eğitim ve test setlerine ayrıldı.")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, verbosity=-1))
])

X_train_processed = model_pipeline.named_steps['preprocessor'].fit_transform(X_train)
X_test_processed = model_pipeline.named_steps['preprocessor'].transform(X_test)
print("Ön işleme (scaling ve encoding) tamamlandı.")

sampler = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed, y_train)
print("Veri, SMOTETomek ile dengelendi.")

model_pipeline.named_steps['classifier'].fit(X_train_resampled, y_train_resampled)
print("Model başarıyla eğitildi.")

probabilities = model_pipeline.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 50)
f1_scores = [f1_score(y_test, probabilities >= t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"En iyi F1-Skoru için optimal eşik (threshold) bulundu: {best_threshold:.4f}")

y_pred_final = (probabilities >= best_threshold).astype(int)
print("\n--- Nihai Model Değerlendirme Raporu ---")
print(classification_report(y_test, y_pred_final, digits=4))
print(f"AUC Skoru: {roc_auc_score(y_test, probabilities):.4f}")

# --- YENİ EKLENTİ: MAHALLE LİSTESİNİ DE KAYDETME ---
final_artifacts = {
    'model_pipeline': model_pipeline,
    'optimal_threshold': best_threshold,
    'neighbourhoods': neighbourhood_list  # Mahalle listesini de sözlüğe ekle
}
joblib.dump(final_artifacts, 'ultimate_noshow_model.joblib')
print("\nModel, eşik değeri ve mahalle listesi 'ultimate_noshow_model.joblib' dosyasına başarıyla kaydedildi.")