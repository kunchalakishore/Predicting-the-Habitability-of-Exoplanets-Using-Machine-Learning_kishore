import joblib
import numpy as np
from app import app, db, Exoplanet



model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("features.pkl")  




with app.app_context():

    planets = Exoplanet.query.all()
    updated = 0
    skipped = 0

    for p in planets:

        values = [
            p.pl_rade,
            p.pl_bmasse,
            p.pl_eqt,
            p.pl_orbper,
            p.st_teff,
            p.st_rad,
        ]

        if any(v is None for v in values):
            skipped += 1
            continue

        X = np.array([values])
        X_scaled = scaler.transform(X)

        score = float(model.predict(X_scaled)[0])
        p.habitability_score = score

        updated += 1

    db.session.commit()


    print(f"Updated planets : {updated}")
    print(f"Skipped planets : {skipped}")
