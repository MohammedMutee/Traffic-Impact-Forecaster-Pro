# 🚦 Traffic Impact Forecaster (Pro)

An AI-augmented urban traffic forecasting command center tailored exclusively for the Bangalore City Limits. 

Built for the Flipkart Round 2 Traffic Management evaluation, this system uses an advanced XGBoost Machine Learning model paired with real-time environmental APIs and NVIDIA NIM Generative AI to act as a virtual Incident Commander.

---

## 🌟 Key Features

*   **Machine Learning Prediction:** Uses an XGBoost classifier (~94% accuracy) to evaluate spatio-temporal data, incident types, and corridor locations to predict Traffic Impact Scores (Low to Extreme).
*   **Live Environmental Telemetry:** Integrates directly with the OpenWeatherMap API for live visibility metrics and the TomTom API for real-time congestion ratios.
*   **AI Command Center:** Powered by NVIDIA NIM (Mixtral 8x7B/DeepSeek). Features a dynamic chat interface to strategize resource deployments and answer unpredictable edge cases.
*   **Geospatial Mapping:** Generates live, interactive Folium maps (CartoDB Dark Matter) dropping glowing pins exactly on the incident coordinates.
*   **Instant PDF Reporting:** One-click generation of fully formatted, professional PDF Incident Reports containing ML metrics and AI strategies.
*   **Fail-Proof Architecture:** Features an automated API-fallback system that ensures 100% LLM uptime by bypassing rate limits seamlessly.

---

## 🛠 Setup & Installation

### 1. Clone the Repository
Ensure all files are stored in the same working directory.

### 2. Install Dependencies
Run the following command to install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. API Keys (Configuration)
All API keys have been centralized for security. Open `config.py` and ensure the keys are populated/active:
*   **OpenWeatherMap:** Live Weather & Visibility.
*   **TomTom:** Live Traffic Congestion flow.
*   **NVIDIA NIM:** Generative AI capabilities.

---

## 🚀 How to Run

1. Open your terminal in the project directory.
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. The high-end dashboard will automatically open in your web browser at `http://localhost:8501`.

---

## 📁 Project Structure

*   `app.py`: The core Streamlit web application and frontend Command Center dashboard.
*   `config.py`: Centralized configuration file storing external API keys.
*   `traffic_impact_model.pkl`: The pre-trained XGBoost prediction engine.
*   `traffic_management_system.py`: The unified backend data processing and modeling pipeline.
*   `prepare_data.py`: Handles raw data cleaning and feature engineering (Categorical Label Encoders).
*   `PROJECT_REPORT.md`: A comprehensive, deep-dive architectural document detailing the A-to-Z development of the ML engine and AI integration.
