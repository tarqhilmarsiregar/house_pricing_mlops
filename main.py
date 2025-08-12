import streamlit as st
import pandas as pd
import tensorflow as tf
import base64
import requests

st.title('House Pricing MLOps - Tarq Hilmar Siregar')

with st.form("house_pricing_form"):
    luas_persegi = st.number_input("Luas Persegi (Square_Footage)", min_value=70, placeholder="Masukkan Luas Persegi (Square_Footage)")

    jumlah_kamar_tidur = st.number_input("Jumlah Kamar Tidur (Num_Bedrooms)", min_value=1, placeholder="Masukkan Jumlah Kamar Tidur (Num_Bedrooms)")

    jumlah_kamar_mandi = st.number_input("Jumlah Kamar Mandi (Num_Bathrooms)", min_value=1, placeholder="Masukkan Jumlah Kamar Mandi (Num_Bathrooms)")

    tahun_dibangun = st.number_input("Tahun Dibangun (Year_Built)", min_value=1950, max_value=2022,placeholder="Masukkan Tahun Dibangun (Year_Built)")

    ukuran_lot = st.number_input("Ukuran Lot (Lot_Size)", min_value=0.0, placeholder="Masukkan Ukuran Lot (Lot_Size)")

    ukuran_garasi = st.number_input("Ukuran Garasi (Garage_Size)", min_value=0, placeholder="Masukkan Ukuran Garasi (Garage_Size)")

    kualitas_lingkungan = st.number_input("Kualitas Lingkungan (Neighborhood_Quality)", min_value=1, max_value=10, placeholder="Masukkan Kualitas Lingkungan (Neighborhood_Quality)")

    predict = st.form_submit_button("Predict")

    if predict:
        # int64 columns
        int_cols = [
            "Square_Footage",
            "Num_Bedrooms",
            "Num_Bathrooms",
            "Year_Built",
            "Garage_Size",
            "Neighborhood_Quality"
        ]

        # float columns
        float_cols = [
            "Lot_Size"
        ]


        def dataframe_to_tfserving_json(df: pd.DataFrame, int_cols=None, float_cols=None):
            """Convert DataFrame ke JSON tf.Example sesuai tipe kolom"""
            instances = []
            int_cols = int_cols or []
            float_cols = float_cols or []
            
            for _, row in df.iterrows():
                features = {}
                for col, val in row.items():
                    if col in int_cols:
                        features[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(val)]))
                    elif col in float_cols:
                        features[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(val)]))
                    else:
                        raise ValueError(f"Kolom {col} tidak ada di mapping tipe data!")
                
                example = tf.train.Example(features=tf.train.Features(feature=features))
                serialized = example.SerializeToString()
                instances.append({"b64": base64.b64encode(serialized).decode("utf-8")})
            
            return {"instances": instances}

        data = {
            "Square_Footage": [luas_persegi],
            "Num_Bedrooms": [jumlah_kamar_tidur],
            "Num_Bathrooms": [jumlah_kamar_mandi],
            "Year_Built": [tahun_dibangun],
            "Lot_Size": [ukuran_lot],
            "Garage_Size": [ukuran_garasi],
            "Neighborhood_Quality": [kualitas_lingkungan]
        }

        df = pd.DataFrame(data)

        int_cols = ["Square_Footage","Num_Bedrooms","Num_Bathrooms","Year_Built","Garage_Size","Neighborhood_Quality"]
        float_cols = ["Lot_Size"]        

        json_data = dataframe_to_tfserving_json(df, int_cols=int_cols, float_cols=float_cols)

        url = "https://house-pricing-tfserving-835674420163.europe-west1.run.app/v1/models/house_pricing_tfserving:predict"

        res = requests.post(url, json=json_data)
        
        st.write(f"Hasil Prediksi (Result): {res.json()["predictions"][0][0]}")

