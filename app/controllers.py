from math import radians, sin, cos, sqrt, atan2
from flask import Blueprint, render_template, request, current_app
from app.models import load_model, load_features, load_poi, compute_nearest_distance
import numpy as np
import pandas as pd
import datetime

main = Blueprint('main', __name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ambil input user
        tanah = float(request.form['luas_tanah'])
        bangunan = float(request.form['luas_bangunan'])

        # Determine which mode was used and get the appropriate location data
        isManualMode = 'mode' in request.form and request.form['mode'] == 'manual'

        if isManualMode:
            # Use values from manual inputs
            lat = float(request.form['lat'])
            lon = float(request.form['lon'])
            kota = request.form.get('kota', 'Jakarta') or 'Jakarta'
            kode_pos = request.form.get('kode_pos', '0') or '0'
        else:
            # Use values from hidden fields
            lat = float(request.form['hidden_lat'])
            lon = float(request.form['hidden_lon'])
            kota = request.form.get('hidden_kota', 'Jakarta') or 'Jakarta'
            kode_pos = request.form.get('hidden_kode_pos', '0') or '0'

        # Load model, features, dan POI
        model = load_model(current_app.config['MODEL_PATH'])
        features = load_features(current_app.config['FEATURES_PATH'])
        poi_df = load_poi(current_app.config['POI_PATTERN'])

        # Filter POIs by type and compute distances
        poi_sekolah = poi_df[poi_df['facility_type'] == 'sekolah']
        poi_rumahsakit = poi_df[poi_df['facility_type'] == 'rumahsakit']
        poi_halte = poi_df[poi_df['facility_type'] == 'halte']
        poi_kampus = poi_df[poi_df['facility_type'] == 'kampus']
        poi_market = poi_df[poi_df['facility_type'] == 'market']
        poi_stasiun = poi_df[poi_df['facility_type'] == 'stasiun']

        jarak_sekolah = compute_nearest_distance(lat, lon, poi_sekolah)
        jarak_rumahsakit = compute_nearest_distance(lat, lon, poi_rumahsakit)
        jarak_halte = compute_nearest_distance(lat, lon, poi_halte)
        jarak_kampus = compute_nearest_distance(lat, lon, poi_kampus)
        jarak_market = compute_nearest_distance(lat, lon, poi_market)
        jarak_stasiun = compute_nearest_distance(lat, lon, poi_stasiun)

        # For backward compatibility with the model, compute the minimum distance
        all_distances = [d for d in [jarak_sekolah, jarak_rumahsakit, jarak_halte,
                                     jarak_kampus, jarak_market, jarak_stasiun] if not np.isnan(d)]
        dist = min(all_distances) if all_distances else np.nan

        # Compute distance from Jakarta (Monas)
        monas_lat = -6.1754
        monas_lon = 106.8272
        jarak_dari_jakarta = haversine_distance(lat, lon, monas_lat, monas_lon)

        years_since_renovation = int(datetime.datetime.now().year - int(request.form.get('tahun_direnovasi', 0) or 0))

        input_vals = {
            'kota': kota,
            'tahun_dibangun': int(request.form['tahun_dibangun']),
            'tahun_direnovasi': int(request.form.get('tahun_direnovasi', 0) or 0),
            'luas_tanah': tanah,
            'luas_bangunan': bangunan,
            'status_tanah': request.form.get('status_tanah') or 'SHM',
            'bentuk_bangunan': request.form['bentuk_bangunan'],
            'listrik': int(request.form['listrik']),
            'pam': 1 if request.form['air'] == 'PAM' else 0,
            'sumur_pompa': 0 if request.form['air'] == 'PAM' else 1,
            'keadaan_lingkungan': request.form['keadaan_lingkungan'],
            'lebar_jalan_(perkerasan)': int(request.form['lebar_jalan']),
            'sarana_transportasi': request.form['sarana_transportasi'],
            'letak_persil': request.form['letak_persil'],
            'lokasi_aset': request.form['lokasi_aset'],
            'kondisi_lingkungan_khusus': request.form['kondisi_lingkungan_khusus'],
            'jarak_ke_stasiun_terdekat': jarak_stasiun,
            'jarak_ke_sekolah_terdekat': jarak_sekolah,
            'jarak_ke_rs_terdekat': jarak_rumahsakit,
            'jarak_ke_market_terdekat': jarak_market,
            'jarak_ke_kampus_terdekat': jarak_kampus,
            'jarak_ke_halte_terdekat': jarak_halte,
            'jarak_dari_jakarta': jarak_dari_jakarta,
            'building_age': int(datetime.datetime.now().year - int(request.form['tahun_dibangun'])),
            'is_renovated': 1 if request.form.get('tahun_direnovasi') else 0,
            'years_since_renovation': 0 if years_since_renovation == datetime.datetime.now().year else years_since_renovation,
            'land_building_ratio': (tanah + 1) / (bangunan + 1)
        }
        print(f"Input vector: {input_vals}")
        # Buat DataFrame dari input
        input_df = pd.DataFrame([input_vals])

        # Prediksi dan penjelasan
        pred = model.predict(input_df)[0]
        pred = np.expm1(pred)
        imps = getattr(model, "feature_importances_", None)
        explanation = ""
        if imps is not None:
            top_idx = np.argsort(imps)[::-1][:3]
            explanation = ', '.join([features[i] for i in top_idx])

        # Calculate additional statistics
        # 1. Harga tanah per meter persegi
        harga_tanah_per_meter = pred * 0.6 / tanah if tanah > 0 else 0  # Assume 60% of property value is land
        
        # 2. Harga bangunan per meter persegi
        harga_bangunan_per_meter = pred * 0.4 / bangunan if bangunan > 0 else 0  # Assume 40% of property value is building
        
        # 3. Prediksi kenaikan 5 tahun (based on various factors)
        # Base appreciation rate
        base_rate = 0.06  # 6% base annual appreciation
        
        # Location factor (closer to Jakarta = higher appreciation)
        location_factor = max(0.02, 0.08 - (jarak_dari_jakarta * 0.005))  # Decreases with distance
        
        # Age factor (newer buildings appreciate more)
        building_age = input_vals['building_age']
        age_factor = max(0.01, 0.05 - (building_age * 0.002))  # Decreases with age
        
        # Renovation factor
        renovation_factor = 0.01 if input_vals['is_renovated'] else 0
        
        # Calculate compound annual growth rate
        annual_growth_rate = base_rate + location_factor + age_factor + renovation_factor
        
        # 5-year prediction with compound interest
        prediksi_5_tahun = pred * ((1 + annual_growth_rate) ** 5)
        kenaikan_5_tahun = prediksi_5_tahun

        # Format the price as currency (Indonesian Rupiah)
        formatted_price = f"Rp{pred:,.2f}".replace(
            ",", "X").replace(".", ",").replace("X", ".")
        
        # Format additional statistics
        formatted_harga_tanah_per_meter = f"Rp{harga_tanah_per_meter:,.0f}".replace(
            ",", "X").replace(".", ",").replace("X", ".")
        
        formatted_harga_bangunan_per_meter = f"Rp{harga_bangunan_per_meter:,.0f}".replace(
            ",", "X").replace(".", ",").replace("X", ".")
        
        formatted_kenaikan_5_tahun = f"Rp{kenaikan_5_tahun:,.0f}".replace(
            ",", "X").replace(".", ",").replace("X", ".")

        return render_template('result.html', price=formatted_price,
                               raw_price=round(pred, 2),
                               distance=round(dist, 2),
                               distances={
                                   'sekolah': round(jarak_sekolah, 2) if not np.isnan(jarak_sekolah) else None,
                                   'rumahsakit': round(jarak_rumahsakit, 2) if not np.isnan(jarak_rumahsakit) else None,
                                   'halte': round(jarak_halte, 2) if not np.isnan(jarak_halte) else None,
                                   'kampus': round(jarak_kampus, 2) if not np.isnan(jarak_kampus) else None,
                                   'market': round(jarak_market, 2) if not np.isnan(jarak_market) else None,
                                   'stasiun': round(jarak_stasiun, 2) if not np.isnan(jarak_stasiun) else None
                               },
                               explanation=explanation,
                               lat=lat, lon=lon,
                                luas_tanah=tanah,
                                luas_bangunan=bangunan,
                                tahun_dibangun=input_vals['tahun_dibangun'],
                                tahun_direnovasi=input_vals['tahun_direnovasi'],
                                lebar_jalan=input_vals['lebar_jalan_(perkerasan)'],
                                kota=kota,
                                kode_pos=kode_pos,
                                keadaan_lingkungan=input_vals['keadaan_lingkungan'],
                                status_tanah=input_vals['status_tanah'],
                                bentuk_bangunan=input_vals['bentuk_bangunan'],
                                # New calculated statistics
                                harga_tanah_per_meter=formatted_harga_tanah_per_meter,
                                harga_bangunan_per_meter=formatted_harga_bangunan_per_meter,
                                prediksi_kenaikan_5_tahun=formatted_kenaikan_5_tahun,
                                annual_growth_rate=f"{annual_growth_rate*100:.1f}%")

    return render_template('index.html')