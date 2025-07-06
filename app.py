from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email
import bcrypt
import pickle
import numpy as np
import pandas as pd


from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
# -------------------- Configuration -------------------- #
import os

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", 'sqlite:///fallback.db')
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "fallback_secret")

db = SQLAlchemy(app)

# -------------------- Database Models -------------------- #
class Registration(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class FQ_data(db.Model):
    id_fq = db.Column(db.Integer, primary_key=True, autoincrement=True)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    moisture = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.Integer, nullable=False)
    crop_type = db.Column(db.Integer, nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    crop_stage = db.Column(db.Integer, nullable=False)
    acres = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    organic_matter = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    season = db.Column(db.Integer, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    phosphorous = db.Column(db.Float, nullable=False)
    f_name = db.Column(db.String(100), nullable=False)
    f_quantity = db.Column(db.Float, nullable=False)

# -------------------- Forms -------------------- #
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# -------------------- Load Models -------------------- #
FN_model = pickle.load(open("FN_model.pkl", "rb"))
FQ_model = pickle.load(open("FQ_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoders.pkl", "rb"))

# -------------------- Auth Routes -------------------- #
@app.route('/registration', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for('register'))

        existing_user = Registration.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "danger")
            return redirect(url_for('register'))

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = Registration(name=name, email=email, password=hashed_password.decode('utf-8'))

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful!", "success")
        return redirect(url_for('acountlogin'))

    return render_template('registration.html')

@app.route('/acountlogin', methods=['GET', 'POST'])
def acountlogin():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        user = Registration.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            session['user_id'] = user.id
            flash("Logged in successfully!", "success")
            return redirect(url_for('home'))
        else:
            flash("Login failed. Check your credentials.", "danger")
            return redirect(url_for('acountlogin'))

    return render_template("acountlogin.html", form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('acountlogin'))

# -------------------- Static Page Routes -------------------- #
@app.route('/')
def home(): return render_template("index.html")

@app.route('/about')
def about(): return render_template("about.html")

@app.route('/fertilizers')
def fertilizers(): return render_template("fertilizers.html")

@app.route('/fert_predict')
def fert_predict(): return render_template("fert_predict.html")

@app.route('/crop_predict')
def crop_predict(): return render_template("crop_predict.html")

@app.route('/crop')
def crop(): return render_template("crop.html")

@app.route('/contact')
def contact(): return render_template("contact.html")

@app.route('/news')
def news(): return render_template("news.html")

@app.route('/communication')
def communucation(): return render_template("communication.html")

@app.route('/sustainable')
def sustainable(): return render_template("sustainable.html")

@app.route('/result')
def result(): return render_template("result.html")

# -------------------- Prediction Route -------------------- #
from flask import request, redirect, render_template, flash, url_for
import numpy as np
import pandas as pd

from flask import request, redirect, render_template, flash, url_for
import numpy as np
import pandas as pd

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            temperature = request.form.get('temperature')
            humidity = request.form.get('humidity')
            moisture = request.form.get('moisture')
            soil_type = request.form.get('soil_type')
            crop_type = request.form.get('crop_type')
            nitrogen = request.form.get('N')
            crop_stage = request.form.get('crop_stage')
            acres = request.form.get('acres')
            ph = request.form.get('ph')
            organic_matter = request.form.get('organic_matter')
            rainfall = request.form.get('rainfall')
            season = request.form.get('season')
            potassium = request.form.get('K')
            phosphorous = request.form.get('P')

            # Check for missing inputs
            fields = [temperature, humidity, moisture, soil_type, crop_type, nitrogen,
                      crop_stage, acres, ph, organic_matter, rainfall, season, potassium, phosphorous]

            if any(field is None or field.strip() == '' for field in fields):
                flash("All fields are required. Please complete the form.", "danger")
                return redirect(url_for('fert_predict'))

            # Convert inputs to correct types
            temperature = float(temperature)
            humidity = float(humidity)
            moisture = float(moisture)
            nitrogen = float(nitrogen)
            acres = float(acres)
            ph = float(ph)
            organic_matter = float(organic_matter)
            rainfall = float(rainfall)
            potassium = float(potassium)
            phosphorous = float(phosphorous)

            # Encode categorical variables using loaded label_encoders
            soil_type_encoded = int(label_encoder['soil_type'].transform([soil_type])[0])
            crop_type_encoded = int(label_encoder['crop_type'].transform([crop_type])[0])
            crop_stage_encoded = int(label_encoder['crop_stage'].transform([crop_stage])[0])
            season_encoded = int(label_encoder['season'].transform([season])[0])

            # Maintain feature order used during model training
            feature_order = ['temparature', 'humidity', 'moisture', 'soil_type', 'crop_type',
                             'nitrogen', 'crop_stage', 'acres', 'pH', 'organic_matter',
                             'rainfall', 'season', 'potassium', 'phosphorous']

            # Build the input features
            input_features = np.array([[temperature, humidity, moisture, soil_type_encoded,
                                        crop_type_encoded, nitrogen, crop_stage_encoded,
                                        acres, ph, organic_matter, rainfall,
                                        season_encoded, potassium, phosphorous]])

            features_df = pd.DataFrame(input_features, columns=feature_order)

            # Get predictions
            prediction = int(FN_model.predict(features_df)[0])
            prediction_q = float(FQ_model.predict(features_df)[0])

            # Fertilizer label mapping
            fertilizer_dict = {
                1: 'Urea',
                2: 'DAP',
                3: '14-35-14',
                4: '28-28',
                5: '20-20'
            }

            p_fertilizer = fertilizer_dict.get(prediction, "Unknown Fertilizer")
            p_quantity = round(prediction_q, 2)

            # Store prediction in DB
            new_data = FQ_data(
                temperature=temperature,
                humidity=humidity,
                moisture=moisture,
                soil_type=soil_type_encoded,
                crop_type=crop_type_encoded,
                nitrogen=nitrogen,
                crop_stage=crop_stage_encoded,
                acres=acres,
                ph=ph,
                organic_matter=organic_matter,
                rainfall=rainfall,
                season=season_encoded,
                potassium=potassium,
                phosphorous=phosphorous,
                f_name=p_fertilizer,
                f_quantity=p_quantity
            )

            db.session.add(new_data)
            db.session.commit()

            # Redirect to result page
            return redirect(url_for('fq_result', f_name=p_fertilizer, f_quantity=p_quantity))

        except Exception as e:
            print("Error during prediction:", e)
            flash("Something went wrong during prediction. Please check your inputs.", "danger")
            return redirect(url_for('fert_predict'))

    # For GET request, render the form page
    return render_template('fert_predict.html')


@app.route('/FQ_predict_result')
def fq_result():
    f_name = request.args.get('f_name')
    f_quantity = request.args.get('f_quantity')
    if f_name and f_quantity:
        return render_template("FQ_predict_result.html", f_name=f_name, f_quantity=f_quantity)
    else:
        flash("Prediction results not found.", "danger")
        return redirect(url_for('fert_predict'))


# -------------------- Run App -------------------- #
if __name__ == '__main__':
    app.run(debug=True)
