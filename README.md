NAFLD AI: AI-Powered Drug Response Prediction

🔖Overview:
NAFLD AI is a full-stack web application designed to accelerate pre-clinical drug discovery for Non-Alcoholic Fatty Liver Disease (NAFLD). The platform allows users to design novel drug candidates by specifying various therapeutic and molecular properties. These candidates are then evaluated by a backend powered by machine learning models to predict their safety and efficacy.

The application features a multi-step workflow where users can design a drug, view its molecular structure, and then simulate its effect on a "Digital Twin." This simulation includes a 12-month clinical trial projection of liver enzyme (ALT) levels and a real-time visualization of the drug's impact at a cellular level on hepatocytes.

🔖Features:
🧪 Drug Design Laboratory: Interactively design drug candidates using sliders to adjust key therapeutic profiles (Anti-Fat, Anti-Inflammatory, Anti-Fibrotic) and molecular properties (Molecular Weight, LogP, H-Bond Donors/Acceptors).

🤖 AI-Powered Prediction: The backend uses a trained XGBoost model to instantly predict the designed drug's safety risk and efficacy.

🔬 Digital Twin Simulation:

Run a 12-month clinical trial simulation using a trained LSTM model to project ALT levels in a virtual patient.

Compare the designed drug's performance against a placebo in a professional clinical report format.

Visualize the drug's impact on fat accumulation and inflammation in a live 2D grid of hepatocyte cells.

📄 Professional Reporting: Generate and save a comprehensive PDF report summarizing the patient profile, clinical trial chart, cellular response, and a plain-language interpretation of the results.

🧬 Multi-View Molecular Visualization: Toggle between a 2D chemical structure diagram and an interactive 3D model for your designed drug.

🔖Technology Stack
Backend
Framework: FastAPI

Server: Uvicorn

Machine Learning:

XGBoost

TensorFlow (Keras)

Scikit-learn

Data Handling: Pandas, NumPy

Model Serialization: Joblib

🔖Frontend
Framework: React (run via Babel in the browser)

Styling: Tailwind CSS

Visualization:

Chart.js (for clinical charts)

Three.js (for 3D molecule rendering)

PDF Generation: jsPDF, html2canvas

🔖Project Structure
/NAFLD_Project
│
├── 📜 main.py                     # The Python backend server (FastAPI)
├── 📜 index.html                  # The single-page React frontend
├── 📜 requirements.txt            # Python dependencies for the backend
│
├── 📦 toxicity_predictor.json     # Trained XGBoost model for safety prediction
├── 📦 lstm_progression_model.keras # Trained LSTM model for ALT progression
└── 📦 alt_scaler.pkl              # Scaler for the LSTM model's data

🚀 Local Setup and Running
To run this project on your local machine, follow these steps.

Pre requisites
Python 3.7+ and Pip installed on your system.

A modern web browser (like Chrome, Firefox, or Edge).

🔖Step 1: Clone or Download the Repository
Download and unzip all the project files into a single folder on your computer.

🔖Step 2: Install Backend Dependencies
Open a Terminal (Command Prompt, PowerShell, or Terminal).

Navigate to the project folder using the cd command.

cd path/to/your/NAFLD_Project

Install the required Python libraries by running:

pip install -r requirements.txt

(This may take a few minutes as it needs to install TensorFlow).

🔖Step 3: Run the Backend Server
In the same terminal (still inside your project folder), start the backend server:

uvicorn main:app --reload

The server is running when you see a line like this:
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

Keep this terminal window open. It is your live backend.

🔖Step 4: Open the Frontend
Navigate to your project folder using your file explorer.

Double-click the index.html file.

This will open the application in your web browser, which will automatically connect to your running backend.

🌐 Deployment Guide
To deploy this application to the web, you need to host the backend and frontend separately.

Part 1: Deploy the Python Backend on Render
Prepare main.py: Ensure the line from tensorflow.keras.models import load_model has been changed to from keras.src.models import load_model.

Push to GitHub: Create a new public GitHub repository and upload all your project files to it.

Create a Render Account: Sign up at render.com, preferably with your GitHub account.

Create a New Web Service:

In Render, click "New +" -> "Web Service".

Connect your GitHub and select your project repository.

Use the following settings:

Runtime: Python 3

Build Command: pip install -r requirements.txt

Start Command: uvicorn main:app --host 0.0.0.0 --port 10000

Deploy: Click "Create Web Service". Wait for the deployment to finish (this can take several minutes).

Copy the URL: Once live, Render will provide a URL (e.g., https://your-backend-name.onrender.com). Copy this URL.

Part 2: Deploy the Frontend to Netlify
Update API URL:

Open your local index.html file.

Find the line: const API_BASE_URL = "http://127.0.0.1:8000";

Replace the local URL with your public Render URL.

Save the file and push this change to your GitHub repository.

Deploy on Netlify:

Sign up at netlify.com.

In the "Sites" section, drag and drop your entire project folder onto the page.

Netlify will deploy your site in seconds and give you a public URL.

You can now share your Netlify URL with anyone to access the live application.
