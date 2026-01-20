from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
import joblib
import os


# App setup
CSV_TOP10_CACHE = []


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///exoplanets.db"
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
@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    importance = model.feature_importances_.tolist()

    return jsonify({
        "features": FEATURES,
        "importance": importance
    })
@app.route("/score_distribution", methods=["GET"])
def score_distribution():
    scores = db.session.query(Exoplanet.habitability_score)\
        .filter(Exoplanet.habitability_score.isnot(None)).all()

    scores = [s[0] for s in scores]

    return jsonify(scores)
@app.route("/correlations", methods=["GET"])
def correlations():
    rows = Exoplanet.query.filter(
        Exoplanet.habitability_score.isnot(None)
    ).all()

    return jsonify({
        "st_teff": [r.st_teff for r in rows],
        "pl_eqt": [r.pl_eqt for r in rows],
        "score": [r.habitability_score for r in rows]
    })
@app.route("/export_top10", methods=["GET"])
def export_top10():
    import pandas as pd

    planets = Exoplanet.query.order_by(
        Exoplanet.habitability_score.desc()
    ).limit(10).all()

    df = pd.DataFrame([{
        "Planet": p.name,
        "Score": p.habitability_score,
        "Radius": p.pl_rade,
        "Mass": p.pl_bmasse
    } for p in planets])

    path = "exoplanets_clean_fill.csv"
    df.to_csv(path, index=False)

    return jsonify({"file": path})
from flask import render_template

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
import pandas as pd
from flask import send_file
from io import BytesIO

@app.route("/export/excel")
def export_excel():

    planets = (
        Exoplanet.query
        .filter(Exoplanet.habitability_score.isnot(None))
        .order_by(Exoplanet.habitability_score.desc())
        .limit(10)
        .all()
    )

    data = [{
        "Rank": i + 1,
        "Planet Name": p.name,
        "Planet Radius": p.pl_rade,
        "Planet Mass": p.pl_bmasse,
        "Equilibrium Temp": p.pl_eqt,
        "Orbital Period": p.pl_orbper,
        "Star Temp": p.st_teff,
        "Star Radius": p.st_rad,
        "Habitability Score": round(p.habitability_score, 4)
    } for i, p in enumerate(planets)]

    df = pd.DataFrame(data)

    output = BytesIO()
    df.to_excel(output, index=False, sheet_name="Top Habitable Planets")
    output.seek(0)

    return send_file(
        output,
        download_name="top_habitable_exoplanets.xlsx",
        as_attachment=True
    )
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

@app.route("/export/pdf")
def export_pdf():

    planets = (
        Exoplanet.query
        .filter(Exoplanet.habitability_score.isnot(None))
        .order_by(Exoplanet.habitability_score.desc())
        .limit(10)
        .all()
    )

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    y = height - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Top 10 Habitable Exoplanets Report")
    y -= 30

    c.setFont("Helvetica", 10)

    for i, p in enumerate(planets, start=1):
        text = (
            f"{i}. {p.name} | "
            f"Score: {round(p.habitability_score,4)} | "
            f"Radius: {p.pl_rade} | "
            f"Mass: {p.pl_bmasse}"
        )
        c.drawString(50, y, text)
        y -= 18

        if y < 60:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 10)

    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        download_name="top_habitable_exoplanets.pdf",
        as_attachment=True
    )
@app.route("/upload_csv_rank", methods=["POST"])
def upload_csv_rank():
    import pandas as pd

    global CSV_TOP10_CACHE

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    df = pd.read_csv(file)

    # Validate required features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return jsonify({
            "error": "Missing required columns",
            "missing_columns": missing
        }), 400

    # Predict
    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    df["habitability_score"] = model.predict(X_scaled)

    
    df_top10 = (
        df.sort_values("habitability_score", ascending=False)
          .head(10)
          .reset_index(drop=True)
    )


    CSV_TOP10_CACHE = [{
        "rank": i + 1,   
        "planet_name": row.get("pl_name", f"Planet {i+1}"),
        "habitability_score": round(row["habitability_score"], 4),
        "pl_rade": row["pl_rade"],
        "pl_bmasse": row["pl_bmasse"]
    } for i, row in df_top10.iterrows()]

    return jsonify(CSV_TOP10_CACHE)

@app.route("/export/csv_excel")
def export_csv_excel():
    from io import BytesIO
    import pandas as pd

    if not CSV_TOP10_CACHE:
        return jsonify({"error": "No CSV ranking available"}), 400

    df = pd.DataFrame(CSV_TOP10_CACHE)

    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(
        output,
        download_name="csv_top10_exoplanets.xlsx",
        as_attachment=True
    )
@app.route("/export/csv_pdf")
def export_csv_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from io import BytesIO

    if not CSV_TOP10_CACHE:
        return jsonify({"error": "No CSV ranking available"}), 400

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Top 10 Habitable Exoplanets (CSV Upload)")
    y -= 30

    c.setFont("Helvetica", 10)
    for p in CSV_TOP10_CACHE:
        c.drawString(
            50, y,
            f"{p['rank']}. {p['planet_name']} | Score: {p['habitability_score']}"
        )
        y -= 18

    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        download_name="csv_top10_exoplanets.pdf",
        as_attachment=True
    )

# Run server

if __name__ == "__main__":
    app.run(debug=True)
