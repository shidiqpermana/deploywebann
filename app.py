from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from miRNA_model import load_model_and_scaler

app = Flask(__name__)
model, scaler = load_model_and_scaler()

# Baca sekali dataset untuk visualisasi
df = pd.read_excel('Gastric.xlsx')
df.columns = df.columns.str.strip()

# Rute form & prediksi
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    FEATURES = ['diana_microt', 'elmmo', 'microcosm', 'miranda', 'mirdb', 'pictar', 'pita', 'targetscan']
    try:
        vals = [float(request.form[f]) for f in FEATURES]
        x = np.array(vals).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0][0]
        return render_template('index.html', prediction=round(float(pred), 4))
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

# Rute visualisasi
@app.route('/visualisasi')
def visualisasi():
    # Sesuaikan kolom dengan FEATURES & TARGET
    FEATURES = ['diana_microt', 'elmmo', 'microcosm', 'miranda', 'mirdb', 'pictar', 'pita', 'targetscan']
    TARGET   = 'all.sum'

    # Pastikan hanya ambil yang benar-benar ada
    feats = [f for f in FEATURES if f in df.columns]
    y_true = df[TARGET].values
    X = df[feats].values
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled).flatten()

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(y_true,   label='Aktual',  marker='o')
    ax.plot(y_pred,   label='Prediksi',marker='x')
    ax.set_xlabel('Data ke-')
    ax.set_ylabel(TARGET)
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return render_template('visualisasi.html', plot_url=img)

if __name__ == '__main__':
    app.run(debug=True)
