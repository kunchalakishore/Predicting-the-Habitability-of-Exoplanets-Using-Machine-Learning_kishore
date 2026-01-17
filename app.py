from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
import joblib
import os


# App setup

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    BASE_DIR, "instance", "exoplanets.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("features.pkl")  


# Database model

class Exoplanet(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), unique=True, nullable=False)

    pl_rade = db.Column(db.Float)
    pl_bmasse = db.Column(db.Float)
    pl_eqt = db.Column(db.Float)
    pl_orbper = db.Column(db.Float)
    st_teff = db.Column(db.Float)
    st_rad = db.Column(db.Float)

    habitability_score = db.Column(db.Float)
    rank = db.Column(db.Integer)



with app.app_context():
    db.create_all()




# Create DB

with app.app_context():
    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    db.create_all()


# Health check

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Exoplanet Habitability API running"
    })


# Add Exoplanet

@app.route("/add_exoplanet", methods=["POST"])
def add_exoplanet():
    data = request.json

    required = [
        "name",
        "pl_rade", "pl_bmasse", "pl_eqt",
        "pl_orbper", "st_teff", "st_rad"
    ]

    for f in required:
        if f not in data:
            return jsonify({"error": f"Missing feature: {f}"}), 400

    if Exoplanet.query.filter_by(name=data["name"]).first():
        return jsonify({"message": "Planet already exists"}), 409

    planet = Exoplanet(
        name=data["name"],
        pl_rade=data["pl_rade"],
        pl_bmasse=data["pl_bmasse"],
        pl_eqt=data["pl_eqt"],
        pl_orbper=data["pl_orbper"],
        st_teff=data["st_teff"],
        st_rad=data["st_rad"]
    )

    db.session.add(planet)
    db.session.commit()

    return jsonify({"message": "Planet added successfully"})


# Predict Habitability (Regression)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    
    try:
        X = np.array([[data[f] for f in FEATURES]])
    except KeyError as e:
        return jsonify({"error": f"Missing feature {e}"}), 400

    
    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled)[0])


    pl_rade = data["pl_rade"]
    pl_bmasse = data["pl_bmasse"]
    pl_eqt = data["pl_eqt"]
    pl_orbper = data["pl_orbper"]
    st_teff = data["st_teff"]
    st_rad = data["st_rad"]

    is_earth_like = (
        0.8 <= pl_rade <= 1.3 and
        0.5 <= pl_bmasse <= 2.0 and
        250 <= pl_eqt <= 320 and
        300 <= pl_orbper <= 430 and
        5000 <= st_teff <= 6200 and
        0.8 <= st_rad <= 1.3
    )

    if is_earth_like:
        prob = max(prob, 0.85)

    habitability = int(prob >= 0.5)

    return jsonify({
        "habitability": habitability,
        "habitability_probability": round(prob, 4),
    })

# Rank Top 10 Habitable Planets

@app.route("/rank", methods=["GET"])
def rank():
    planets = (
        Exoplanet.query
        .filter(Exoplanet.habitability_score.isnot(None))
        .order_by(Exoplanet.rank.asc())
        .limit(10)
        .all()
    )

    return jsonify([
        {
            "rank": p.rank,
            "planet_name": p.name,
            "habitability_score": round(p.habitability_score, 10)
        }
        for p in planets
    ])
@app.route("/secure_predict", methods=["POST"])
def secure_predict():

    
    # AUTH CHECK

    token = request.headers.get("Authorization")

    if token != "Bearer SECRET123":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()

    
    # FEATURE VALIDATION
    
    try:
        X = np.array([[data[f] for f in FEATURES]])
    except KeyError as e:
        return jsonify({
            "error": f"Missing feature: {str(e)}",
            "required_features": FEATURES
        }), 400


    X_scaled = scaler.transform(X)
    score = float(model.predict(X_scaled)[0])

    return jsonify({
        "status": "success",
        "secure": True,
        "habitability_score": round(score, 6)
    })

# Run server

if __name__ == "__main__":
    app.run(debug=True)
