from flask import Blueprint, render_template, request, current_app
from app.models import load_model, load_features, load_poi, compute_nearest_distance
import numpy as np
import pandas as pd
import datetime

main = Blueprint('main', __name__)


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

        input_vals = {
            'luas_tanah': tanah,
            'luas_bangunan': bangunan,
            'tahun_dibangun': int(request.form['tahun_dibangun']),
            'tahun_direnovasi': int(request.form.get('tahun_direnovasi', 0) or 0),
            'lebar_jalan_(perkerasan)': int(request.form['lebar_jalan']),
            'jarak_ke_stasiun_terdekat': jarak_stasiun,
            'jarak_ke_sekolah_terdekat': jarak_sekolah,
            'jarak_ke_rs_terdekat': jarak_rumahsakit,
            'jarak_ke_market_terdekat': jarak_market,
            'jarak_ke_kampus_terdekat': jarak_kampus,
            'jarak_ke_halte_terdekat': jarak_halte,
            'latitude': lat,
            'longitude': lon,
            'status_tanah': request.form.get('status_tanah') or 'SHM',
            'kota': kota,
            'kode_pos': kode_pos,
            'bentuk_bangunan': request.form['bentuk_bangunan'],
            'kondisi_lingkungan_khusus': request.form['kondisi_lingkungan_khusus'],
            'letak_persil': request.form['letak_persil'],
            'lokasi_aset': request.form['lokasi_aset'],
            'umur_bangunan': int(datetime.datetime.now().year - int(request.form['tahun_dibangun'])),
            'waktu_sejak_renovasi': int(datetime.datetime.now().year - int(request.form.get('tahun_direnovasi', 0) or 0)),
            'luas_per_km_stasiun': tanah / (jarak_stasiun + 1),
            'luas_per_km_sekolah': tanah / (jarak_sekolah + 1),
            'building_to_land_ratio': bangunan / tanah,
            'jarak_total': jarak_stasiun + jarak_sekolah + jarak_rumahsakit + jarak_market + jarak_kampus + jarak_halte
        }
        print(f"Input vector: {input_vals}")
        # Buat DataFrame dari input
        input_df = pd.DataFrame([input_vals])

        # Periksa apakah semua fitur yang dibutuhkan tersedia
        missing_features = set(features) - set(input_df.columns)
        if missing_features:
            raise ValueError(
                f"Fitur yang dibutuhkan tidak tersedia: {missing_features}")

        # Seleksi fitur yang dibutuhkan dengan urutan yang sesuai
        input_df = input_df[features]

        # Prediksi dan penjelasan
        pred = model.predict(input_df)[0]
        pred = np.expm1(pred)
        imps = getattr(model, "feature_importances_", None)
        explanation = ""
        if imps is not None:
            top_idx = np.argsort(imps)[::-1][:3]
            explanation = ', '.join([features[i] for i in top_idx])

        # Format the price as currency (Indonesian Rupiah)
        formatted_price = f"Rp {pred:,.2f}".replace(
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
                               lat=lat, lon=lon)

    return render_template('index.html')
