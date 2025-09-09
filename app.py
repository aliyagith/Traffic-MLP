from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd
from joblib import load
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import plotly.express as px
import numpy as np

app = Flask(__name__)
app.secret_key = 'traffic_secret_key' #for flask message
app.config['SECRET_KEY']= 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ------------------ Auth Helper ------------------
def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please login to continue.')
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

# Load models lazily to keep app startup simple
_density_model = None
_incident_model = None

def _ensure_sklearn_pickle_compat():
    """Add missing symbols expected by older/newer scikit-learn pickles.

    Some saved pipelines reference private classes like
    sklearn.compose._column_transformer._RemainderColsList which may not exist
    in your installed scikit-learn version. We provide a minimal shim so the
    pickle can be deserialized.
    """
    try:
        from sklearn.compose import _column_transformer as ct_mod  # type: ignore
        if not hasattr(ct_mod, '_RemainderColsList'):
            class _RemainderColsList(list):  # minimal placeholder to satisfy unpickling
                pass
            ct_mod._RemainderColsList = _RemainderColsList  # type: ignore[attr-defined]
    except Exception:
        # Best-effort shim; if this fails we'll surface the original load error
        pass

def get_density_model():
    global _density_model
    if _density_model is None:
        model_path = os.path.join('Traffic_Model', 'traffic_density_xgb_pipeline.pkl')
        try:
            _density_model = load(model_path)
        except Exception:
            _ensure_sklearn_pickle_compat()
            _density_model = load(model_path)
    return _density_model

def get_incident_model():
    global _incident_model
    if _incident_model is None:
        model_path = os.path.join('Traffic_Model', 'incident_xgb_pipeline.pkl')
        try:
            _incident_model = load(model_path)
        except Exception:
            _ensure_sklearn_pickle_compat()
            _incident_model = load(model_path)
    return _incident_model


def build_density_explanation(row: dict, pred: float) -> list:
    reasons = []
    # Peak hour
    if row.get('Is Peak Hour') in (1, '1', 'on', True):
        reasons.append('Peak hour typically increases congestion and density.')
    # Random event
    try:
        if int(row.get('Random Event Occurred', 0)) == 1:
            reasons.append('A random event was reported, which tends to spike density.')
    except Exception:
        pass
    # Weather
    weather = (row.get('Weather') or '').lower()
    if weather in ['rainy', 'snowy', 'electromagnetic storm', 'solar flare']:
        reasons.append(f"Weather condition '{row.get('Weather')}' generally slows traffic and raises density.")
    elif weather == 'clear':
        reasons.append("Clear weather is associated with lower density.")
    # Speed
    try:
        spd = float(row.get('Speed', 0))
        if spd <= 40:
            reasons.append('Lower observed speed suggests heavier congestion.')
        elif spd >= 90:
            reasons.append('Higher speed indicates freer flow, reducing density.')
    except Exception:
        pass
    # Hour bands
    try:
        hr = int(row.get('Hour Of Day', 0))
        if 7 <= hr <= 10 or 17 <= hr <= 20:
            reasons.append('Typical rush hours (commute times) elevate density.')
    except Exception:
        pass
    # City/Vehicle/Economy mention
    if row.get('Economic Condition') in ['Recession']:
        reasons.append('During recession, demand patterns vary; some corridors may see elevated density.')
    if not reasons:
        reasons.append('Conditions suggest moderate flow; model combined all inputs for this estimate.')
    reasons.insert(0, f"Estimated density: {pred:.4f} (lower means freer flow, higher means more congestion).")
    return reasons


def build_incident_explanation(row: dict, proba: float, pred: int) -> list:
    reasons = [
        f"Incident likelihood: {proba*100:.2f}% → {'Likely' if pred==1 else 'Unlikely'}"
    ]
    # Peak hour
    if row.get('Is Peak Hour') in (1, '1', 'on', True):
        reasons.append('Peak hour increases exposure and minor disruptions likelihood.')
    # Weather
    weather = (row.get('Weather') or '').lower()
    if weather in ['rainy', 'snowy', 'electromagnetic storm', 'solar flare']:
        reasons.append(f"Adverse weather ('{row.get('Weather')}') raises incident risk.")
    # Speed extremes
    try:
        spd = float(row.get('Speed', 0))
        if spd >= 100:
            reasons.append('Very high speeds can correlate with higher incident risk.')
        elif spd <= 25:
            reasons.append('Very low speeds may reflect disruptive conditions in progress.')
    except Exception:
        pass
    # Hour bands
    try:
        hr = int(row.get('Hour Of Day', 0))
        if 22 <= hr or hr <= 5:
            reasons.append('Late-night/early-morning hours can have elevated risk on some corridors.')
    except Exception:
        pass
    if not reasons:
        reasons.append('Model combined all inputs to estimate the probability.')
    return reasons

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict")
@login_required
def predict():
    return render_template('predict.html')

# About route
@app.route("/about")
def about():
    return render_template('About.html')

@app.route("/login")
def login():
    # GET renders template; POST handled in /auth/login
    return render_template('login.html', error=None)


# ------------------ Auth Helpers & DB ------------------
def get_db():
    conn = sqlite3.connect(os.path.join(os.getcwd(), 'traffic_auth.db'))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()


def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please login to continue.')
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


# @app.before_first_request
# def _bootstrap():
#     init_db()


@app.context_processor
def inject_user():
    return {
        'current_user': {
            'id': session.get('user_id'),
            'name': session.get('user_name'),
            'email': session.get('user_email')
        }
    }


# ------------------ Auth Routes ------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        if not name or not email or not password:
            return render_template('register.html', error='All fields are required.')
        if password != confirm:
            return render_template('register.html', error='Passwords do not match.')
        pw_hash = generate_password_hash(password)
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute('INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)', (name, email, pw_hash))
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Email already registered.')
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
    return render_template('register.html', error=None)


@app.route('/auth/login', methods=['POST'])
def auth_login():
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT id, name, email, password_hash FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    conn.close()
    if not row or not check_password_hash(row['password_hash'], password):
        return render_template('login.html', error='Invalid email or password')
    session['user_id'] = row['id']
    session['user_name'] = row['name']
    session['user_email'] = row['email']
    flash('Logged in successfully!')
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('index'))

 
        
# -------- Density Prediction --------
@app.route('/predict/density', methods=['GET', 'POST'])
@login_required
def predict_density():
    if request.method == 'POST':
        try:
            form = request.form
            # Build a single-row DataFrame matching training feature names
            row = {
                'City': form.get('City'),
                'Vehicle Type': form.get('Vehicle Type'),
                'Weather': form.get('Weather'),
                'Economic Condition': form.get('Economic Condition'),
                'Day Of Week': form.get('Day Of Week'),
                'Hour Of Day': int(form.get('Hour Of Day', 0)),
                'Speed': float(form.get('Speed', 0)),
                'Is Peak Hour': 1 if form.get('Is Peak Hour') == 'on' else 0,
                'Random Event Occurred': int(form.get('Random Event Occurred', 0)),
                'Energy Consumption': float(form.get('Energy Consumption', 0.0)),
            }
            df = pd.DataFrame([row])
            model = get_density_model()
            pred = float(model.predict(df)[0])
            explanation = build_density_explanation(row, pred)
            return render_template('density_predict.html', prediction=pred, form_data=row, explanation=explanation)
        except Exception as e:
            return render_template('density_predict.html', prediction=None, form_data=form.to_dict(), error=str(e))
    # GET
    return render_template('density_predict.html', prediction=None, form_data={})


# -------- Incident/Disruption Classification --------
@app.route('/predict/incident', methods=['GET', 'POST'])
@login_required
def predict_incident():
    if request.method == 'POST':
        try:
            form = request.form
            # Features exclude target and leakage columns (Traffic Density, Energy Consumption)
            row = {
                'City': form.get('City'),
                'Vehicle Type': form.get('Vehicle Type'),
                'Weather': form.get('Weather'),
                'Economic Condition': form.get('Economic Condition'),
                'Day Of Week': form.get('Day Of Week'),
                'Hour Of Day': int(form.get('Hour Of Day', 0)),
                'Speed': float(form.get('Speed', 0)),
                'Is Peak Hour': 1 if form.get('Is Peak Hour') == 'on' else 0,
            }
            # Backward-compat: if current model pipeline expects leakage columns, provide neutral defaults
            row.setdefault('Energy Consumption', 0.0)
            row.setdefault('Traffic Density', 0.0)
            df = pd.DataFrame([row])
            model = get_incident_model()
            proba = float(model.predict_proba(df)[0, 1])
            pred = int(proba >= 0.5)
            explanation = build_incident_explanation(row, proba, pred)
            return render_template('incident_predict.html', prediction=pred, probability=proba, form_data=row, explanation=explanation)
        except Exception as e:
            return render_template('incident_predict.html', prediction=None, probability=None, form_data=form.to_dict(), error=str(e))
    # GET
    return render_template('incident_predict.html', prediction=None, probability=None, form_data={})


# -------- Density Dashboard --------
@app.route('/dashboard/density')
@login_required
def density_dashboard():
    try:
        # Load data once per request (could be cached if needed)
        csv_path = os.path.join(os.getcwd(), 'futuristic_city_traffic.csv')
        df = pd.read_csv(csv_path)

        # 1) Density vs Hour of Day (line)
        line_data = df.groupby('Hour Of Day')['Traffic Density'].mean().reset_index()
        fig_line = px.line(line_data, x='Hour Of Day', y='Traffic Density',
                        title='Traffic Density vs Hour of Day (Peak Hour Trends)')
        line_html = fig_line.to_html(full_html=False, include_plotlyjs='cdn')

        # 2) Heatmap: Density Hour × Day
        heatmap_data = df.groupby(['Day Of Week','Hour Of Day'])['Traffic Density'].mean().reset_index()
        fig_heat = px.density_heatmap(heatmap_data, x='Hour Of Day', y='Day Of Week', z='Traffic Density',
                                    color_continuous_scale='Viridis', title='Heatmap of Traffic Density (Hour × Day)')
        heat_html = fig_heat.to_html(full_html=False, include_plotlyjs=False)

        # 3) Bar: Avg Density per City
        bar_data = df.groupby('City')['Traffic Density'].mean().reset_index()
        fig_bar = px.bar(bar_data, x='City', y='Traffic Density', title='Average Traffic Density per City',
                        color='Traffic Density', color_continuous_scale='Blues')
        bar_html = fig_bar.to_html(full_html=False, include_plotlyjs=False)

        # 4) Box: Density by Day of Week
        fig_box = px.box(df, x='Day Of Week', y='Traffic Density', title='Distribution of Traffic Density by Day of Week')
        box_html = fig_box.to_html(full_html=False, include_plotlyjs=False)

        # 5) Scatter: Speed vs Density
        scatter_sample = df.sample(n=min(5000, len(df)), random_state=42)
        fig_scatter = px.scatter(scatter_sample, x='Traffic Density', y='Speed',
                                title='Speed vs Traffic Density (Congestion Effect)', opacity=0.6)
        scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

        # KPIs
        pk = line_data.loc[line_data['Traffic Density'].idxmax()]
        lw = line_data.loc[line_data['Traffic Density'].idxmin()]
        kpis = {
            'peak_hour': int(pk['Hour Of Day']),
            'peak_density': float(pk['Traffic Density']),
            'low_hour': int(lw['Hour Of Day']),
            'low_density': float(lw['Traffic Density'])
        }

        return render_template('density_dashboard.html',
                            line_html=line_html,
                            heat_html=heat_html,
                            bar_html=bar_html,
                            box_html=box_html,
                            scatter_html=scatter_html,
                            kpis=kpis)
    except Exception as e:
        flash(f'Dashboard error: {e}')
        return redirect(url_for('index'))


# -------- Incident Dashboard --------
@app.route('/dashboard/incident')
@login_required
def incident_dashboard():
    try:
        csv_path = os.path.join(os.getcwd(), 'futuristic_city_traffic.csv')
        df = pd.read_csv(csv_path)

        # 1) Avg Speed: Event vs No Event (bar)
        speed_event = df.groupby('Random Event Occurred')['Speed'].mean().reset_index()
        speed_event['Event'] = speed_event['Random Event Occurred'].map({0: 'No Event', 1: 'Event'})
        fig_speed = px.bar(speed_event, x='Event', y='Speed', title='Average Speed: Event vs No Event',
                        color_discrete_sequence=['#1f77b4'])
        speed_html = fig_speed.to_html(full_html=False, include_plotlyjs='cdn')

        # 2) Scatter: Density vs Speed (events highlighted)
        scatter_sample = df.sample(n=min(5000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            scatter_sample, x='Traffic Density', y='Speed',
            color=scatter_sample['Random Event Occurred'].map({0: 'No Event', 1: 'Event'}),
            title='Traffic Density vs Speed (Events Highlighted)', opacity=0.6,
            color_discrete_sequence=['#2ca02c', '#d62728']
        )
        scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

        # 3) Heatmap: Event frequency City × Day
        event_city_day = df.groupby(['City', 'Day Of Week'])['Random Event Occurred'].sum().reset_index()
        fig_ev_heat = px.density_heatmap(event_city_day, x='Day Of Week', y='City', z='Random Event Occurred',
                                        title='Frequency of Random Events (City × Day of Week)', color_continuous_scale='Blues')
        ev_heat_html = fig_ev_heat.to_html(full_html=False, include_plotlyjs=False)

        # 4) Box: Energy vs Event
        df['Event Label'] = df['Random Event Occurred'].map({0: 'No Event', 1: 'Event'})
        fig_energy_box = px.box(df, x='Event Label', y='Energy Consumption',
                                title='Energy Consumption Distribution: Event vs No Event',
                                color='Event Label', color_discrete_sequence=['#2ca02c', '#d62728'])
        energy_box_html = fig_energy_box.to_html(full_html=False, include_plotlyjs=False)

        # 5) Line: Density over time with Events highlighted
        density_event = df.groupby(['Hour Of Day', 'Random Event Occurred'])['Traffic Density'].mean().reset_index()
        density_event['Event Label'] = density_event['Random Event Occurred'].map({0: 'No Event', 1: 'Event'})
        fig_time = px.line(density_event, x='Hour Of Day', y='Traffic Density', color='Event Label',
                        title='Traffic Density over Time with Events Highlighted', markers=True,
                        color_discrete_sequence=['#2ca02c', '#d62728'])
        time_html = fig_time.to_html(full_html=False, include_plotlyjs=False)

        # KPIs/Insights
        kpis = {}
        if set(['Event', 'Speed']).issubset(set(speed_event.columns)) and speed_event['Event'].nunique() == 2:
            gap = speed_event.set_index('Event').loc['No Event', 'Speed'] - speed_event.set_index('Event').loc['Event', 'Speed']
            kpis['speed_drop_event'] = float(gap)

        # top hotspots
        top_cells = event_city_day.sort_values('Random Event Occurred', ascending=False).head(5)
        hotspots = [f"{r['City']} × {r['Day Of Week']}: {int(r['Random Event Occurred'])}" for _, r in top_cells.iterrows()]

        return render_template('incident_dashboard.html',
                            speed_html=speed_html,
                            scatter_html=scatter_html,
                            ev_heat_html=ev_heat_html,
                            energy_box_html=energy_box_html,
                            time_html=time_html,
                            kpis=kpis,
                            hotspots=hotspots)
    except Exception as e:
        flash(f'Incident dashboard error: {e}')
        return redirect(url_for('index'))


if __name__ == "__main__":
    if not os.path.exists('users.db'):
        with app.app_context():
            db.create_all()
    app.run(debug=True)