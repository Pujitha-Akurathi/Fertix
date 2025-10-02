🌱 Fertix – Sustainable Fertilizer Usage Optimizer

Fertix is an AI-powered project developed as part of Smart India Hackathon 2024 (College Level). The system helps farmers optimize fertilizer usage for higher yield while promoting sustainable farming practices.

📌 Features

🔍 AI-based Recommendation Engine – Suggests suitable fertilizers and crops by analyzing soil & weather datasets.

💬 Multilingual Chatbot – NLP-based chatbot offering personalized farming support in multiple languages.

⚡ Flask APIs + MySQL – Backend APIs integrated with database for real-time recommendations.

🌐 Farmer-Friendly Interface – Easy-to-use interface with interactive support.

🌎 Sustainability Focus – Encourages eco-friendly practices by reducing fertilizer overuse.

🛠️ Tech Stack

Programming Languages: Python, SQL, JavaScript

Backend: Flask, MySQL

Frontend: HTML, CSS, React.js (basic interface)

AI/ML: Scikit-learn, NLP models

Other Tools: Pandas, NumPy, OpenCV (if image processing used), Power BI (for visualization)

📂 Project Structure
Fertix/
│── backend/
│   ├── app.py             # Flask API
│   ├── models/            # ML Models
│   ├── database.sql       # MySQL Schema
│── frontend/
│   ├── index.html
│   ├── static/            # CSS, JS files
│── chatbot/
│   ├── nlp_chatbot.py     # Multilingual Chatbot
│── datasets/
│   ├── soil_data.csv
│   ├── weather_data.csv
│── README.md

🚀 How to Run

Clone Repository

git clone https://github.com/your-username/fertix.git
cd fertix


Backend Setup

cd backend
pip install -r requirements.txt
python app.py


Frontend Setup

Open index.html in browser

Or serve with a simple server:

python -m http.server 8000


Chatbot Setup

cd chatbot
python nlp_chatbot.py

📊 Results

✅ Optimized fertilizer usage by recommending the right combination of fertilizers.

✅ Improved crop yield predictions using AI models.

✅ Enhanced farmer support via real-time multilingual chatbot.

👩‍💻 Team

Developed as part of Smart India Hackathon 2024 by Team Fertix.
