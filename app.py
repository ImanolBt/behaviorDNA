from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import requests
import base64
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
from io import BytesIO

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = "8653872557:AAH1aeiptrE3MVKbPSYXsx17UNTAnPhqVuc"
TELEGRAM_CHAT_ID   = "8606708090"

models        = {}
profiles      = {}
training_data = {}
alert_log     = []

USERS = ['Dr. Ramirez', 'Dr. Lopez', 'Enf. Garcia', 'Dr. Mendoza', 'Tec. Vargas']

def extract_features(intervals):
    arr = np.array(intervals)
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.median(arr))
    ]

def train_model(user):
    data = training_data.get(user, [])
    if len(data) < 3:
        return False
    X, y = [], []
    for f in data:
        X.append(f); y.append(1)
    base_mean = np.mean([f[0] for f in data])
    for factor in [0.25, 0.35, 0.45, 0.55, 2.0, 2.5, 3.0, 3.5, 4.0]:
        fake = [base_mean*factor, base_mean*0.2, base_mean*factor*0.5, base_mean*factor*1.5, base_mean*factor]
        X.append(fake); y.append(0)
    model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    model.fit(X, y)
    models[user] = model
    all_means = [f[0] for f in data]
    profiles[user] = {
        'mean': float(np.mean(all_means)),
        'std':  float(np.std(all_means)) if np.std(all_means) > 0 else 30.0
    }
    return True

def send_telegram(message, photo_b64=None):
    try:
        if photo_b64:
            img_data = base64.b64decode(photo_b64.split(',')[1] if ',' in photo_b64 else photo_b64)
            url   = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': ('intruder.jpg', img_data, 'image/jpeg')}
            data  = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'HTML'}
            requests.post(url, files=files, data=data, timeout=10)
        else:
            url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
            requests.post(url, json=data, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/users', methods=['GET'])
def get_users():
    result = []
    for u in USERS:
        result.append({
            'name':    u,
            'trained': u in models,
            'samples': len(training_data.get(u, [])),
            'mean':    round(profiles[u]['mean'], 1) if u in profiles else None,
        })
    return jsonify({'users': result})

@app.route('/train', methods=['POST'])
def train():
    data      = request.json
    intervals = data.get('intervals', [])
    user      = data.get('user', 'Dr. Ramirez')

    if len(intervals) < 3:
        return jsonify({'ok': False, 'msg': 'Muy pocos datos'})

    features = extract_features(intervals)
    if user not in training_data:
        training_data[user] = []
    training_data[user].append(features)
    total = len(training_data[user])

    ready = False
    if total >= 3:
        ready = train_model(user)

    return jsonify({
        'ok':           True,
        'user':         user,
        'total_samples': total,
        'ready':        ready,
        'profile_mean': round(profiles[user]['mean'], 1) if user in profiles else None,
        'profile_std':  round(profiles[user]['std'], 1)  if user in profiles else None,
    })

@app.route('/verify', methods=['POST'])
def verify():
    data        = request.json
    intervals   = data.get('intervals', [])
    user        = data.get('user', 'Dr. Ramirez')
    is_intruder = data.get('intruder', False)
    photo_b64   = data.get('photo', None)

    if user not in models:
        return jsonify({'ok': False, 'msg': 'Usuario no entrenado'})
    if len(intervals) < 3:
        return jsonify({'ok': False, 'msg': 'Muy pocos datos'})

    features    = extract_features(intervals)
    features_np = np.array(features).reshape(1, -1)
    model       = models[user]
    profile     = profiles[user]

    prediction  = model.predict(features_np)[0]
    proba       = model.predict_proba(features_np)[0]
    confidence  = round(float(max(proba)) * 100, 1)

    test_mean  = features[0]
    test_std   = features[1]
    z_score    = abs(test_mean - profile['mean']) / max(profile['std'], 30)
    similarity = max(0, round(100 - z_score * 22))
    granted    = bool(prediction == 1)

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    if is_intruder or not granted:
        alert_log.append({
            'timestamp':  timestamp,
            'user':       user,
            'test_mean':  round(test_mean, 1),
            'z_score':    round(z_score, 2),
            'similarity': similarity,
        })
        msg = (
            f"ALERTA DE SEGURIDAD - BehaviorDNA\n\n"
            f"Hospital Central - Latacunga\n"
            f"{timestamp}\n\n"
            f"INTENTO DE ACCESO NO AUTORIZADO\n"
            f"Usuario objetivo: {user}\n\n"
            f"Similitud: {similarity}%\n"
            f"Desviacion: {round(z_score,2)} sigma\n"
            f"Media de escritura: {round(test_mean,1)} ms\n\n"
            f"Acceso bloqueado automaticamente."
        )
        send_telegram(msg, photo_b64)

    return jsonify({
        'ok':           True,
        'granted':      granted,
        'confidence':   confidence,
        'similarity':   similarity,
        'z_score':      round(z_score, 2),
        'test_mean':    round(test_mean, 1),
        'test_std':     round(test_std, 1),
        'profile_mean': round(profile['mean'], 1),
        'profile_std':  round(profile['std'], 1),
        'timestamp':    timestamp,
    })

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data   = request.json
    buffer = BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    story  = []

    title_style = ParagraphStyle('title', fontSize=20, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#cc0022'), spaceAfter=6)
    sub_style = ParagraphStyle('sub', fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#555555'), spaceAfter=20)
    section_style = ParagraphStyle('section', fontSize=12, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#111111'), spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle('body', fontSize=9, fontName='Helvetica',
        textColor=colors.HexColor('#333333'), spaceAfter=4, leading=14)
    alert_style = ParagraphStyle('alert', fontSize=13, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#cc0022'), spaceAfter=8)

    story.append(Paragraph("HOSPITAL CENTRAL - LATACUNGA", sub_style))
    story.append(Paragraph("REPORTE FORENSE DE SEGURIDAD BIOMETRICA", title_style))
    story.append(Paragraph(f"BehaviorDNA - {data.get('timestamp','')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#cc0022')))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("ACCESO NO AUTORIZADO DETECTADO", alert_style))
    story.append(Paragraph(
        "El sistema BehaviorDNA detecto un intento de acceso con credenciales validas "
        "pero patron conductual no reconocido. Acceso bloqueado automaticamente.", body_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("DATOS BIOMETRICOS", section_style))

    table_data = [
        ['PARAMETRO', 'VALOR', 'REFERENCIA'],
        ['Media de escritura (ms)', str(data.get('test_mean','')), str(data.get('profile_mean',''))],
        ['Desviacion estandar (ms)', str(data.get('test_std','')), str(data.get('profile_std',''))],
        ['Puntuacion Z', str(data.get('z_score','')), '< 1.8 (umbral)'],
        ['Similitud conductual', str(data.get('similarity',''))+'%', '> 70% (requerido)'],
        ['Resultado', 'ACCESO DENEGADO', 'ACCESO CONCEDIDO'],
    ]
    table = Table(table_data, colWidths=[6*cm, 4.5*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),  (-1,0),  colors.HexColor('#111111')),
        ('TEXTCOLOR',     (0,0),  (-1,0),  colors.white),
        ('FONTNAME',      (0,0),  (-1,0),  'Helvetica-Bold'),
        ('BACKGROUND',    (0,-1), (-1,-1), colors.HexColor('#ffeeee')),
        ('TEXTCOLOR',     (0,-1), (-1,-1), colors.HexColor('#cc0022')),
        ('FONTNAME',      (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('ALIGN',         (0,0),  (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0),  (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),  (-1,-2), [colors.HexColor('#f9f9f9'), colors.white]),
        ('GRID',          (0,0),  (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('FONTSIZE',      (0,0),  (-1,-1), 9),
        ('ROWHEIGHT',     (0,0),  (-1,-1), 0.7*cm),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4*cm))

    z = float(data.get('z_score', 0))
    nivel = "CRITICO" if z > 3 else ("ALTO" if z > 2 else "MODERADO")
    story.append(Paragraph("ANALISIS", section_style))
    story.append(Paragraph(f"Nivel de amenaza: {nivel}. Desviacion de {data.get('z_score','')} sigma, similitud {data.get('similarity','')}%.", body_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#dddddd')))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("BehaviorDNA - Unidad Educativa Oxford - Latacunga, Ecuador.",
        ParagraphStyle('footer', fontSize=7, fontName='Helvetica', textColor=colors.HexColor('#999999'))))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f"reporte_forense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                     mimetype='application/pdf')

@app.route('/reset', methods=['POST'])
def reset():
    global models, profiles, training_data, alert_log
    data = request.json
    user = data.get('user', None) if data else None
    if user:
        models.pop(user, None)
        profiles.pop(user, None)
        training_data.pop(user, None)
    else:
        models={}; profiles={}; training_data={}; alert_log=[]
    return jsonify({'ok': True})

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)