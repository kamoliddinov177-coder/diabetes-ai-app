import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Sahifa sozlamalari
st.set_page_config(
    page_title="Diabet AI — Erta Diagnostika", 
    page_icon="🩸",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Sarlavha va kirish
st.title("🩸 Diabet Xavfini Erta Aniqlovchi AI Tizimi")
st.markdown("""
Ushbu sun'iy intellekt modeli metabolik ko'rsatkichlar hamda insulin rezistentligini (**HOMA-IR**) tahlil qilib, 
diabet rivojlanish xavfini hisoblaydi va har bir ko'rsatkichning ta'sirini **SHAP (Explainable AI)** orqali tushuntiradi.
""")
st.divider()

# 2. Modelni o'qitish va keshga olish
@st.cache_resource
def load_and_train():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=cols)

    # Imputatsiya (Nollarni median bilan to'ldirish)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    for col in zero_cols:
        df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))

    # Feature Engineering (HOMA-IR indeksi)
    df['HOMA_IR'] = (df['Insulin'] * (df['Glucose'] / 18.0)) / 22.5

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns

model, explainer, feature_names = load_and_train()

# 3. Yon panel (Sidebar) - Bemor ko'rsatkichlari
st.sidebar.header("📋 Bemor Ko'rsatkichlari")
st.sidebar.write("Laboratoriya va fiziologik ma'lumotlarni kiriting:")

glucose = st.sidebar.slider("Glukoza (och qoringa, mg/dL)", 50, 250, 115)
insulin = st.sidebar.slider("Insulin (och qoringa, µU/mL)", 15, 400, 75)
bmi = st.sidebar.slider("Tana vazni indeksi (BMI, kg/m²)", 15.0, 55.0, 26.5)
age = st.sidebar.slider("Yoshi", 20, 90, 35)
blood_pressure = st.sidebar.slider("Qon bosimi (diastolik, mmHg)", 40, 140, 72)
skin_thickness = st.sidebar.slider("Teri burmasi qalinligi (mm)", 10, 99, 22)
pregnancies = st.sidebar.number_input("Homiladorliklar soni", 0, 20, 1)
dpf = st.sidebar.number_input("Genetik moyillik (Pedigree Function)", 0.08, 2.50, 0.45, step=0.01)

# HOMA-IR avtomatik hisoblash
homa_ir = (insulin * (glucose / 18.0)) / 22.5

# Input Dataframe yaratish
input_df = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age, homa_ir
]], columns=feature_names)

# 4. Asosiy Bo'lim - Diagnostika va SHAP
col1, col2 = st.columns([1, 1.2])

prob = model.predict_proba(input_df)[0][1]
risk_percentage = prob * 100

with col1:
    st.subheader("📊 Diagnostika Xulosasi")
    st.metric(label="Diabet Rivojlanish Xavfi", value=f"{risk_percentage:.1f}%")
    
    st.write(f"**HOMA-IR Indeksi:** `{homa_ir:.2f}`")
    if homa_ir > 2.7:
        st.caption("⚠️ *Yuqori insulin rezistentligi aniqlandi (Me'yor: < 2.0)*")
    elif homa_ir >= 2.0:
        st.caption("⚡ *Chegaraviy insulin rezistentligi (Me'yor: < 2.0)*")
    else:
        st.caption("✅ *Insulin sezgirligi me'yorda (Me'yor: < 2.0)*")

    st.divider()

    if risk_percentage >= 50:
        st.error("🚨 **Yuqori xavf guruhi!** Metabolik ko'rsatkichlarda jiddiy og'ishlar bor.")
    elif risk_percentage >= 25:
        st.warning("⚠️ **O'rta xavf guruhi (Pre-diabet ehtimoli).** Nazorat va profilaktika talab etiladi.")
    else:
        st.success("✅ **Past xavf guruhi.** Ko'rsatkichlar barqaror va me'yorda.")

with col2:
    st.subheader("🧠 SHAP Tahlili (Qaror Sababi)")
    st.write("Qaysi ko'rsatkich xavfni **oshirgan** (qizil) yoki **kamaytirgan** (ko'k):")

    shap_values = explainer(input_df)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

st.divider()

# 5. Avtomatik Tibbiy va Parhez Maslahatlari
st.subheader("💡 Avtomatik Tibbiy va Parhez Maslahatlari")

if risk_percentage >= 50:
    col_med, col_diet = st.columns(2)
    with col_med:
        st.markdown("### 🏥 Tibbiy Tavsiyalar")
        st.error("""
        1. **Endokrinolog Ko'rigi:** Tez orada mutaxassis ko'rigidan o'ting.
        2. **Qo'shimcha Analizlar:**
           - **HbA1c** (Glikatsiyalangan gemoglobin) – oxirgi 3 oylik glukoza darajasi.
           - **OGTT** (Oral glukoza bag'rikenglik testi).
        3. **Qon Bosimi va Buyrak Tahlili:** Haftada 2 marta arterial bosimni nazorat qiling.
        """)

    with col_diet:
        st.markdown("### 🥗 Parhez va Turmush Tarzi")
        st.warning("""
        1. **Uglevodlarni Chegaralash:** Shakar, shirin ichimliklar, oq non va hamir ovqatlarni butunlay to'xtating.
        2. **Past Glikemik Indeks (GI):** Sabzavotlar, dukkakli mahsulotlar hamda grechka/suli yormalariga o'ting.
        3. **Jismoniy Faollik:** Kuniga kamida 45 daqiqa tez yurish yoki aerobik mashqlar (insulin sezgirligini oshiradi).
        """)

elif risk_percentage >= 25:
    col_med, col_diet = st.columns(2)
    with col_med:
        st.markdown("### 🏥 Tibbiy Tavsiyalar")
        st.warning("""
        1. **Profilaktik Ko'rik:** Har 6 oyda och qoringa glukoza va insulin tahlilini topshirib turing.
        2. **Vazn Nazorati:** Tana vaznini 5–7% ga kamaytirish diabet xavfini 50% dan ortiqqa qisqartiradi.
        """)

    with col_diet:
        st.markdown("### 🥗 Parhez va Turmush Tarzi")
        st.info("""
        1. **Tovoq Qoidasi (Plate Method):** Tovoqning 50% qismini sabzavotlar, 25% oqsil (go'sht/baliq), 25% murakkab uglevodlar tashkil etsin.
        2. **Kechki Ovqat:** Uxlashdan kamida 3 soat oldin ovqatlanishni to'xtating.
        3. **Kunlik Yurish:** Kuniga kamida 8,000–10,000 qadam bosishga harakat qiling.
        """)

else:
    col_med, col_diet = st.columns(2)
    with col_med:
        st.markdown("### 🏥 Tibbiy Tavsiyalar")
        st.success("""
        1. **Yillik Tekshiruv:** Yiliga 1 marta umumiy profilaktik qon tahlillarini topshirish yetarli.
        2. **Uyqu Rejimi:** Kuniga 7-8 soat sifatli uyqu metabolik salomatlikni saqlaydi.
        """)

    with col_diet:
        st.markdown("### 🥗 Parhez va Turmush Tarzi")
        st.success("""
        1. **Sog'lom Ovqatlanish:** Qand va qayta ishlangan mahsulotlarni me'yorda tuting.
        2. **Suv Balansi:** Kuniga 1.5–2 litr toza suv ichishni odat qiling.
        3. **Faol Hordiq:** Sport va faol turmush tarzini davom ettiring.
        """)
