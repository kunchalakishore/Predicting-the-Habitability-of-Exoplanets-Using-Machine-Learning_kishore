from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# DATABASE CONFIG 
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///exoplanets.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# LOAD MODEL 
model = joblib.load("model.pkl")

FEATURES = [
    "pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper",
    "st_teff", "st_rad", "st_lum", "sy_dist"
]

# DATABASE MODEL 
class Exoplanet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

    pl_rade = db.Column(db.Float)
    pl_bmasse = db.Column(db.Float)
    pl_eqt = db.Column(db.Float)
    pl_orbper = db.Column(db.Float)
    st_teff = db.Column(db.Float)
    st_rad = db.Column(db.Float)
    st_lum = db.Column(db.Float)
    sy_dist = db.Column(db.Float)

    habitability = db.Column(db.Integer)
    habitability_probability = db.Column(db.Float)

# CREATE DB 
with app.app_context():
    db.create_all()

# HEALTH CHECK 
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Flask Exoplanet API running"
    })

#  ADD EXOPLANET 
@app.route("/add_exoplanet", methods=["POST"])
def add_exoplanet():
    data = request.get_json()

    # Check duplicate
    existing = Exoplanet.query.filter_by(name=data["name"]).first()
    if existing:
        return jsonify({
            "status": "error",
            "message": "Planet already exists in database"
        }), 409

    planet = Exoplanet(
        name=data["name"],
        pl_rade=data["pl_rade"],
        pl_bmasse=data["pl_bmasse"],
        pl_eqt=data["pl_eqt"],
        pl_orbper=data["pl_orbper"],
        st_teff=data["st_teff"],
        st_rad=data["st_rad"],
        st_lum=data["st_lum"],
        sy_dist=data["sy_dist"]
    )

    db.session.add(planet)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Exoplanet stored successfully"
    }), 201


# PREDICT 
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    X = np.array([[data[f] for f in FEATURES]])
    probability = model.predict_proba(X)[0][1]
    habitability = int(probability >= 0.5)

    planet = Exoplanet.query.filter_by(name=data["name"]).first()
    if planet:
        planet.habitability = habitability
        planet.habitability_probability = float(probability)
        db.session.commit()

    return jsonify({
        "planet_name": data["name"],
        "habitability": habitability,
        "habitability_probability": round(probability, 3)
    })

# RANK 
@app.route("/rank", methods=["GET"])
def rank():
    planets = Exoplanet.query.filter(
        Exoplanet.habitability_probability.isnot(None)
    ).order_by(
        Exoplanet.habitability_probability.desc()
    ).limit(10).all()

    return jsonify([
        {
            "rank": i + 1,
            "planet_name": p.name,
            "habitability": p.habitability,
            "habitability_probability": round(p.habitability_probability, 10)
        }
        for i, p in enumerate(planets)
    ])



# SECURE PREDICT 
@app.route("/secure_predict", methods=["POST"])
def secure_predict():
    token = request.headers.get("Authorization")

    if token != "Bearer SECRET123":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    X = np.array([[data[f] for f in FEATURES]])
    probability = model.predict_proba(X)[0][1]

    return jsonify({
        "status": "success",
        "habitability_probability": round(float(probability), 3)
    })

# RUN 
if __name__ == "__main__":
    app.run(debug=True)
