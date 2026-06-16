import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import requests
from openai import OpenAI
from fpdf import FPDF
import folium
from streamlit_folium import st_folium
import config

# Must be the first Streamlit command
st.set_page_config(page_title="Traffic Impact Forecaster", layout="wide")

class TrafficReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(30, 41, 59)
        self.cell(0, 10, 'OFFICIAL TRAFFIC IMPACT REPORT', border=0, align='C', fill=False, new_x="LMARGIN", new_y="NEXT")
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Generated automatically by Traffic Forecaster', align='C')

def create_pdf_report(data, ai_insight=""):
    pdf = TrafficReportPDF()
    pdf.add_page()
    
    # Event Parameters Box
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 10, ' Event Parameters', border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(50, 8, 'Event Type:', border=1)
    pdf.cell(140, 8, str(data.get('event_type', '')), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 8, 'Corridor:', border=1)
    pdf.cell(140, 8, str(data.get('corridor', '')), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 8, 'Coordinates:', border=1)
    pdf.cell(140, 8, f"{data.get('lat', '')}, {data.get('lon', '')}", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Telemetry
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, ' Live Telemetry Data', border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(50, 8, 'Weather:', border=1)
    pdf.cell(45, 8, str(data.get('weather_desc', '')), border=1)
    pdf.cell(50, 8, 'Congestion Ratio:', border=1)
    pdf.cell(45, 8, f"{data.get('current_congestion_ratio', 1.0):.2f}x", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Deployment Plan
    plan = data.get('plan', {})
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    pdf.cell(0, 10, f" IMPACT LEVEL: {plan.get('Level', '')} (Conf: {data.get('confidence', 0):.1f}%)", border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(255, 255, 255)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(50, 10, 'Priority Tier:', border='L')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(140, 10, str(plan.get('Priority', '')), border='R', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(50, 10, 'Manpower:', border='L')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(140, 10, str(plan.get('Manpower', '')), border='R', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(50, 10, 'Equipment:', border='L')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(140, 10, str(plan.get('Barricades', '')), border='R', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(50, 10, 'Diversion Strategy:', border='L,B')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(140, 10, str(plan.get('Diversion', '')), border='R,B', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # AI Summary
    if ai_insight:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(0, 10, ' Strategic AI Summary', border=0, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font('Helvetica', '', 10)
        
        # Sanitize common Unicode characters that are unsupported by base Helvetica
        safe_insight = ai_insight.replace('**', '').replace('—', '-').replace('–', '-').replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, safe_insight, border=1)
        
    return bytes(pdf.output())

def call_ai_with_fallback(client, messages):
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-flash",
            messages=messages,
            temperature=0.6,
            max_tokens=1024,
            extra_body={"chat_template_kwargs":{"thinking":False}},
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Fallback to mixtral
        try:
            completion = client.chat.completions.create(
                model="mistralai/mixtral-8x7b-instruct-v0.1",
                messages=messages,
                temperature=0.6,
                max_tokens=1024,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e2:
            raise Exception("Both DeepSeek and Mixtral models failed.")

# Enhanced Custom CSS for Perfect Formatting
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #FFFFFF;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        transition: all 0.3s ease;
        padding: 0.6rem 2rem;
        font-weight: 500;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #2563EB 100%);
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 12px;
        background-color: #1E293B;
        border: 1px solid #334155;
        margin-top: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .priority-p1 { border-left: 6px solid #EF4444; }
    .priority-p2 { border-left: 6px solid #F97316; }
    .priority-p3 { border-left: 6px solid #EAB308; }
    .priority-p4 { border-left: 6px solid #22C55E; }
    
    div[data-baseweb="select"] > div {
        background-color: #1E293B;
        border-color: #334155;
        border-radius: 6px;
    }
    .stNumberInput input, .stDateInput input, .stTimeInput input {
        background-color: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 6px;
    }
    /* Chat bubbles */
    .stChatMessage {
        background-color: #1E293B !important;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load Model and Encoders
@st.cache_resource
def load_model():
    try:
        data = joblib.load('traffic_impact_model.pkl')
        return data['model'], data['features'], data['encoders']
    except Exception as e:
        st.error("Model file not found. Please ensure 'traffic_impact_model.pkl' exists.")
        return None, None, None

model, features, encoders = load_model()

def get_action_plan(impact_score):
    recommendations = {
        0: {
            "Level": "Low",
            "Manpower": "1 Traffic Officer (Monitoring)",
            "Barricades": "0 required",
            "Diversion": "None",
            "Priority": "P4 (Routine)"
        },
        1: {
            "Level": "Medium",
            "Manpower": "2 Officers (On-site)",
            "Barricades": "2-4 (Local containment)",
            "Diversion": "Local advisory",
            "Priority": "P3 (Standard Response)"
        },
        2: {
            "Level": "High",
            "Manpower": "4-6 Officers + 1 Inspector",
            "Barricades": "10+ (Lane management)",
            "Diversion": "Active Local Diversion",
            "Priority": "P2 (High Priority)"
        },
        3: {
            "Level": "Extreme",
            "Manpower": "12+ Officers + ACP Oversight",
            "Barricades": "30+ (Complete perimeter)",
            "Diversion": "Systemic Grid Diversion Plan",
            "Priority": "P1 (Emergency)"
        }
    }
    return recommendations.get(impact_score, recommendations[0])

st.title("Traffic Impact Forecaster (Pro)")
st.markdown("<p style='color: #94A3B8; margin-bottom: 0.5rem; font-size: 1.1rem;'>Predict urban traffic impact utilizing live environmental, social, and transit feeds.</p>", unsafe_allow_html=True)
st.markdown("<p style='color: #6366F1; margin-bottom: 2rem; font-size: 0.9rem; font-weight: 500; letter-spacing: 0.5px;'><span style='background-color: rgba(99, 102, 241, 0.1); padding: 0.3rem 0.8rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);'>📍 Exclusively Optimized for Bangalore City Limits</span></p>", unsafe_allow_html=True)

if model and encoders:
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.subheader("Event Parameters")
        
        # Group inputs
        c1, c2 = st.columns(2)
        with c1:
            date_input = st.date_input("Event Date")
            event_type = st.selectbox("Event Type", encoders['event_type'].classes_)
            corridor = st.selectbox("Corridor", encoders['corridor'].classes_)
            police_station = st.selectbox("Police Station", encoders['police_station'].classes_)
        with c2:
            time_input = st.time_input("Event Time")
            event_cause = st.selectbox("Event Cause", encoders['event_cause'].classes_)
            zone = st.selectbox("Zone", encoders['zone'].classes_)
            veh_type = st.selectbox("Vehicle Type (if applicable)", encoders['veh_type'].classes_)
            
        c3, c4 = st.columns(2)
        with c3:
            lat = st.number_input("Latitude", value=12.9716, format="%.4f")
        with c4:
            lon = st.number_input("Longitude", value=77.5946, format="%.4f")
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Live Forecast"):
            # --- REAL API INTEGRATION ---
            with st.spinner("Connecting to OpenWeatherMap and TomTom APIs..."):
                OWM_API_KEY = config.OWM_API_KEY
                TOMTOM_API_KEY = config.TOMTOM_API_KEY
                
                is_raining = 0
                visibility_index = 10
                weather_desc = "Clear"
                try:
                    w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}"
                    w_res = requests.get(w_url, timeout=3)
                    if w_res.status_code == 200:
                        w_data = w_res.json()
                        weather_main = w_data['weather'][0]['main'].lower()
                        weather_desc = weather_main.title()
                        if 'rain' in weather_main or 'drizzle' in weather_main or 'thunderstorm' in weather_main:
                            is_raining = 1
                        vis_m = w_data.get('visibility', 10000)
                        visibility_index = min(10, int(vis_m / 1000))
                except Exception as e:
                    pass
                
                current_congestion_ratio = 1.0
                try:
                    t_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={TOMTOM_API_KEY}&point={lat},{lon}"
                    t_res = requests.get(t_url, timeout=3)
                    if t_res.status_code == 200:
                        t_data = t_res.json()
                        flow = t_data.get('flowSegmentData', {})
                        cs = flow.get('currentSpeed', 1)
                        ffs = flow.get('freeFlowSpeed', 1)
                        if cs > 0:
                            ratio = ffs / cs
                            current_congestion_ratio = np.clip(ratio, 0.5, 3.0)
                except Exception as e:
                    pass
                
                tweet_volume_surge = 0
                transit_disrupted = 0
            
            # Prepare Input Data
            dt = pd.to_datetime(f"{date_input} {time_input}")
            
            def safe_encode(le, val):
                if val in le.classes_: return le.transform([val])[0]
                if 'unknown' in le.classes_: return le.transform(['unknown'])[0]
                return 0

            input_dict = {
                'hour': dt.hour,
                'day_of_week': dt.dayofweek,
                'is_weekend': 1 if dt.dayofweek in [5, 6] else 0,
                'month': dt.month,
                'latitude': lat,
                'longitude': lon,
                'is_raining': is_raining,
                'visibility_index': visibility_index,
                'current_congestion_ratio': current_congestion_ratio,
                'tweet_volume_surge': tweet_volume_surge,
                'transit_disrupted': transit_disrupted,
                'event_type_enc': safe_encode(encoders['event_type'], event_type),
                'event_cause_enc': safe_encode(encoders['event_cause'], event_cause),
                'corridor_enc': safe_encode(encoders['corridor'], corridor),
                'police_station_enc': safe_encode(encoders['police_station'], police_station),
                'veh_type_enc': safe_encode(encoders['veh_type'], veh_type),
                'zone_enc': safe_encode(encoders['zone'], zone)
            }
            
            input_df = pd.DataFrame([input_dict])[features]
            prediction = model.predict(input_df)[0]
            confidence = model.predict_proba(input_df)[0][prediction] * 100
            plan = get_action_plan(prediction)
            
            st.session_state.forecast_data = {
                'plan': plan,
                'confidence': confidence,
                'weather_desc': weather_desc,
                'visibility_index': visibility_index,
                'is_raining': is_raining,
                'current_congestion_ratio': current_congestion_ratio,
                'event_type': event_type,
                'event_cause': event_cause,
                'corridor': corridor,
                'lat': lat,
                'lon': lon
            }
            
            # Generate Initial AI Insight
            with st.spinner("Command Center AI is analyzing the situation..."):
                nv_api_key = config.NVIDIA_API_KEY
                client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nv_api_key, timeout=15.0)
                
                system_prompt = f"You are an elite Traffic Operations Commander AI assistant. You have deep expertise in urban traffic management, resource allocation, and emergency diversion tactics. Be professional, concise, and do not use emojis."
                ai_prompt = f"An event of type '{event_type}' caused by '{event_cause}' is occurring at Corridor '{corridor}'. Our ML model predicted an Impact Level of '{plan['Level']}' ({confidence:.1f}% confidence). Current weather is {weather_desc} and live traffic congestion is {current_congestion_ratio:.2f}x the baseline. The initial system deployment plan recommends {plan['Manpower']}, {plan['Barricades']} barricades, and {plan['Diversion']}. Provide a concise 2-paragraph strategic summary analyzing this situation and confirming the deployment."
                
                st.session_state.chat_history = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ai_prompt}
                ]
                
                try:
                    ai_content = call_ai_with_fallback(client, st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_content})
                except Exception as e:
                    fallback_msg = "Command Center AI Analysis timed out or failed. You can try chatting below to reconnect."
                    st.session_state.chat_history.append({"role": "assistant", "content": fallback_msg})
                    
            st.rerun()
        
        if st.session_state.forecast_data:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h4 style='margin-top: 0; margin-bottom: 0.5rem; color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;'>Incident Location Map</h4>", unsafe_allow_html=True)
                map_data = st.session_state.forecast_data
                m = folium.Map(location=[map_data['lat'], map_data['lon']], zoom_start=15, tiles="CartoDB dark_matter")
                folium.Marker(
                    [map_data['lat'], map_data['lon']],
                    popup=map_data['corridor'],
                    tooltip=f"Impact: {map_data['plan']['Level']}",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)
                st_folium(m, height=250, use_container_width=True, returned_objects=[])
        
    with col2:
        if st.session_state.forecast_data:
            data = st.session_state.forecast_data
            plan = data['plan']
            priority_class = f"priority-{plan['Priority'][:2].lower()}"
            
            # Dashboard
            st.markdown(f"""
                <div class="metric-box">
                    <h4 style="margin-top: 0; margin-bottom: 1rem; color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Live External Feeds</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem;">Weather Condition</span><br>
                            <span style="font-size: 1.05rem; font-weight: 500; color: #F8FAFC;">{data['weather_desc']} (Vis: {data['visibility_index']}/10)</span>
                        </div>
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem;">Social Alert Volume</span><br>
                            <span style="font-size: 1.05rem; font-weight: 500; color: #475569;">Offline (Baseline)</span>
                        </div>
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem;">Current Congestion</span><br>
                            <span style="font-size: 1.05rem; font-weight: 500; color: #F8FAFC;">{data['current_congestion_ratio']:.2f}x Baseline</span>
                        </div>
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem;">Public Transit</span><br>
                            <span style="font-size: 1.05rem; font-weight: 500; color: #475569;">Offline (Baseline)</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="prediction-box {priority_class}">
                    <h2 style="margin-top: 0; color: #FFFFFF; font-weight: 600;">Impact Level: {plan['Level']}</h2>
                    <p style="color: #94A3B8; font-size: 0.9rem;">Algorithm Confidence: <span style="color: #38BDF8;">{data['confidence']:.1f}%</span></p>
                    <hr style="border-color: #334155; margin: 1.5rem 0;">
                    <div style="display: flex; flex-direction: column; gap: 1.2rem;">
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Priority Tier</span><br>
                            <span style="font-size: 1.15rem; color: #F8FAFC; font-weight: 500;">{plan['Priority']}</span>
                        </div>
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Manpower Required</span><br>
                            <span style="font-size: 1.15rem; color: #F8FAFC; font-weight: 500;">{plan['Manpower']}</span>
                        </div>
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Equipment</span><br>
                            <span style="font-size: 1.15rem; color: #F8FAFC; font-weight: 500;">{plan['Barricades']}</span>
                        </div>
                        <div>
                            <span style="color: #64748B; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Diversion Strategy</span><br>
                            <span style="font-size: 1.15rem; color: #F8FAFC; font-weight: 500;">{plan['Diversion']}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # --- PDF DOWNLOAD ---
            st.markdown("<br>", unsafe_allow_html=True)
            ai_text = ""
            if len(st.session_state.chat_history) > 2:
                ai_text = st.session_state.chat_history[2]['content']
            
            try:
                pdf_bytes = create_pdf_report(data, ai_text)
                st.download_button(
                    label="📄 Download Official Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"Traffic_Report_{data.get('corridor', 'export').replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
            
        else:
            st.markdown("""
                <div style="padding: 3rem; border-radius: 12px; border: 2px dashed #334155; text-align: center; background-color: #1E293B; margin-top: 0.5rem; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <div style="font-size: 2.5rem; color: #475569; margin-bottom: 1rem;">📊</div>
                    <h3 style="color: #94A3B8; font-weight: 400;">Awaiting Input</h3>
                    <p style="color: #64748B;">Configure the event parameters on the left and click "Generate Live Forecast" to view the optimal deployment strategy alongside live data.</p>
                </div>
            """, unsafe_allow_html=True)

# --- BOTTOM SECTION: AI CHAT ---
if st.session_state.forecast_data:
    st.markdown("<hr style='border-color: #1E293B; margin: 3rem 0; border-width: 2px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFFFFF; text-align: center; margin-bottom: 2rem; font-size: 1.5rem;'>Command Center AI</h3>", unsafe_allow_html=True)
    
    chat_col1, chat_col2, chat_col3 = st.columns([1, 4, 1])
    
    with chat_col2:
        chat_container = st.container(height=400, border=True)
        
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "system":
                    continue
                if msg["role"] == "user" and msg == st.session_state.chat_history[1]:
                    # Don't show the initial hidden system prompt we passed as user
                    continue
                
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        if prompt := st.chat_input("Discuss unpredictable problems or ask for alternatives..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("AI is thinking..."):
                        nv_api_key = config.NVIDIA_API_KEY
                        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nv_api_key, timeout=15.0)
                        try:
                            response_content = call_ai_with_fallback(client, st.session_state.chat_history)
                            st.markdown(response_content)
                            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
                        except Exception as e:
                            st.error(f"Failed to connect to AI: {e}")
