# 🌍 Exoplanet Habitability Prediction System

An AI-powered web application that predicts, ranks, and visualizes the habitability of exoplanets using machine learning.

---

##  Problem Statement
Identifying potentially habitable exoplanets from astronomical datasets is complex. This project uses ML to automate habitability scoring and ranking.

---

##  Technologies Used
- Python, Flask
- Scikit-learn, XGBoost
- SQLite
- HTML, CSS, JavaScript
- Plotly.js
- Pandas
- Render (Deployment)

---

## Machine Learning
- Model: XGBoost Regressor
- Features:
  - Planet radius
  - Planet mass
  - Equilibrium temperature
  - Orbital period
  - Star temperature
  - Star radius
- Output: Habitability score (0–1)

---

## Features
✔ Single planet prediction  
✔ Bulk CSV upload & ranking  
✔ Top 10 habitable planets  
✔ Feature importance visualization  
✔ Score distribution & correlations  
✔ Excel & PDF export  
✔ Dynamic star background UI  



---

## CSV Upload Format
```csv
pl_rade,pl_bmasse,pl_eqt,pl_orbper,st_teff,st_rad,pl_name
