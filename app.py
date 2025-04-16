import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===========================================
# INITIALIZATION & CONFIGURATION
# ===========================================

class MaintenanceSystem:
    def __init__(self):
        # Email configuration from environment variables
        self.email_config = {
            'outlook': {
                'sender': os.getenv('OUTLOOK_EMAIL'),
                'password': os.getenv('OUTLOOK_PASSWORD'),
                'smtp_server': 'smtp.office365.com',
                'smtp_port': 587,
                'active': os.getenv('USE_OUTLOOK', 'False').lower() == 'true'
            },
            'gmail': {
                'sender': os.getenv('GMAIL_EMAIL'),
                'password': os.getenv('GMAIL_PASSWORD'),
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'active': os.getenv('USE_GMAIL', 'False').lower() == 'true'
            },
            'primary_service': os.getenv('PRIMARY_EMAIL_SERVICE', 'outlook'),
            'recipients': os.getenv('ALERT_RECIPIENTS', 'maintenance-team@your-company.com').split(',')
        }
        
        # Priority configuration
        self.priority_config = {
            'Critical': {'threshold': 85, 'color': '#d9534f', 'response_time': '1 hour'},
            'High': {'threshold': 70, 'color': '#f0ad4e', 'response_time': '4 hours'},
            'Medium': {'threshold': 50, 'color': '#5bc0de', 'response_time': '24 hours'},
            'Low': {'threshold': 0, 'color': '#5cb85c', 'response_time': '1 week'}
        }
        
        # Initialize databases
        self.init_databases()

    def init_databases(self):
        """Initialize all required databases with error handling"""
        try:
            # Prediction database
            pred_conn = sqlite3.connect('turbine_monitoring.db')
            pred_conn.execute('''CREATE TABLE IF NOT EXISTS predictions
                               (timestamp TEXT, turbine_id TEXT, failure_probability REAL, 
                                features TEXT, notes TEXT)''')
            pred_conn.close()
            
            # Tickets database
            ticket_conn = sqlite3.connect('turbine_maintenance.db')
            ticket_conn.execute('''CREATE TABLE IF NOT EXISTS tickets
                                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  turbine_id TEXT,
                                  issue TEXT,
                                  severity TEXT,
                                  probability REAL,
                                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                  status TEXT DEFAULT 'Open',
                                  resolved_at TIMESTAMP,
                                  assigned_to TEXT)''')
            ticket_conn.close()
        except sqlite3.Error as e:
            st.error(f"Database initialization failed: {str(e)}")
            raise

    def log_prediction(self, turbine_id, probability, features, notes=""):
        """Log prediction to database with error handling"""
        conn = None
        try:
            conn = sqlite3.connect('turbine_monitoring.db')
            c = conn.cursor()
            c.execute('''INSERT INTO predictions VALUES (?, ?, ?, ?, ?)''',
                     (datetime.now().isoformat(), turbine_id, probability,
                      json.dumps(features), notes))
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Failed to log prediction: {str(e)}")
        finally:
            if conn:
                conn.close()

    def create_ticket(self, turbine_id, issue, probability):
        """Create maintenance ticket with automated alerts"""
        # Determine severity
        severity = next(
            (level for level, config in self.priority_config.items() 
             if probability >= config['threshold']),
            'Low'
        )
        
        # Create database record
        conn = None
        try:
            conn = sqlite3.connect('turbine_maintenance.db')
            c = conn.cursor()
            c.execute('''INSERT INTO tickets 
                        (turbine_id, issue, severity, probability)
                        VALUES (?, ?, ?, ?)''',
                     (turbine_id, issue, severity, probability))
            ticket_id = c.lastrowid
            conn.commit()
            
            # Send alert
            if self.email_config[self.email_config['primary_service']]['active']:
                self.send_alert(ticket_id, turbine_id, issue, severity, probability)
            
            return ticket_id
        except sqlite3.Error as e:
            st.error(f"Failed to create ticket: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()

    def send_alert(self, ticket_id, turbine_id, issue, severity, probability):
        """Send email alert with fallback mechanism"""
        services = [self.email_config['primary_service']]
        if self.email_config['gmail']['active'] and self.email_config['primary_service'] != 'gmail':
            services.append('gmail')
        
        for service in services:
            try:
                config = self.email_config[service]
                msg = MIMEMultipart()
                msg['From'] = config['sender']
                msg['To'] = ", ".join(self.email_config['recipients'])
                msg['Subject'] = f"⚠️ {severity} Alert - {turbine_id} (Ticket #{ticket_id})"
                
                # HTML email body
                html = f"""
                <html>
                  <body style="font-family: Arial, sans-serif;">
                    <div style="border:1px solid #ddd; border-radius:5px; max-width:600px;">
                      <div style="background:{self.priority_config[severity]['color']}; 
                                  color:white; padding:15px; border-radius:5px 5px 0 0;">
                        <h2 style="margin:0;">Maintenance Ticket #{ticket_id}</h2>
                      </div>
                      <div style="padding:20px;">
                        <table style="width:100%; margin-bottom:15px;">
                          <tr><td><strong>Turbine ID:</strong></td><td>{turbine_id}</td></tr>
                          <tr><td><strong>Severity:</strong></td><td>{severity}</td></tr>
                          <tr><td><strong>Probability:</strong></td><td>{probability:.1f}%</td></tr>
                          <tr><td><strong>Response Time:</strong></td><td>{self.priority_config[severity]['response_time']}</td></tr>
                        </table>
                        <div style="background:#f5f5f5; padding:15px; border-radius:3px;">
                          <h3 style="margin-top:0;">Issue Details</h3>
                          <p>{issue}</p>
                        </div>
                      </div>
                    </div>
                  </body>
                </html>
                """
                msg.attach(MIMEText(html, 'html'))
                
                with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                    server.starttls()
                    server.login(config['sender'], config['password'])
                    server.send_message(msg)
                return True
            except Exception as e:
                st.error(f"Failed to send email via {service}: {str(e)}")
                continue
        return False

    def get_open_tickets(self):
        """Retrieve all open tickets"""
        conn = None
        try:
            conn = sqlite3.connect('turbine_maintenance.db')
            df = pd.read_sql("SELECT * FROM tickets WHERE status = 'Open' ORDER BY created_at DESC", conn)
            return df
        except sqlite3.Error as e:
            st.error(f"Failed to retrieve tickets: {str(e)}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    def resolve_ticket(self, ticket_id, assigned_to=None):
        """Mark ticket as resolved"""
        conn = None
        try:
            conn = sqlite3.connect('turbine_maintenance.db')
            c = conn.cursor()
            if assigned_to:
                c.execute('''UPDATE tickets 
                            SET status = 'Resolved', resolved_at = CURRENT_TIMESTAMP, assigned_to = ?
                            WHERE id = ?''', (assigned_to, ticket_id))
            else:
                c.execute('''UPDATE tickets 
                            SET status = 'Resolved', resolved_at = CURRENT_TIMESTAMP 
                            WHERE id = ?''', (ticket_id,))
            conn.commit()
            return True
        except sqlite3.Error as e:
            st.error(f"Failed to resolve ticket: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()

# Initialize maintenance system
try:
    maint_system = MaintenanceSystem()
except Exception as e:
    st.error(f"Failed to initialize maintenance system: {str(e)}")
    st.stop()

# ===========================================
# FEATURE CONFIGURATION
# ===========================================

# Define all expected features in the EXACT order the model expects them
FEATURE_ORDER = [
    'rotor_speed',
    'generator_speed',
    'power_output',
    'blade_pitch_angle',
    'gearbox_temp',
    'generator_temp',
    'vibration',
    'oil_pressure',
    'wind_speed',
    'ambient_temp',
    'humidity',
    'time_since_maintenance'
]

# Define realistic ranges for each feature for simulation
FEATURE_RANGES = {
    'rotor_speed': (10.0, 20.0),
    'generator_speed': (1400.0, 1600.0),
    'power_output': (1500.0, 2500.0),
    'blade_pitch_angle': (0.0, 10.0),
    'gearbox_temp': (50.0, 100.0),
    'generator_temp': (60.0, 110.0),
    'vibration': (1.0, 5.0),
    'oil_pressure': (80.0, 120.0),
    'wind_speed': (5.0, 15.0),
    'ambient_temp': (10.0, 30.0),
    'humidity': (40.0, 80.0),
    'time_since_maintenance': (0, 365)
}

# ===========================================
# STREAMLIT APPLICATION
# ===========================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Wind Turbine Maintenance", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🌬️ Wind Turbine Predictive Maintenance System")
    
    # Load ML model with error handling
    model = None
    try:
        model = joblib.load('best_model.pkl')
        # Verify model has correct number of features
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(FEATURE_ORDER):
            st.error(f"Model expects {model.n_features_in_} features but we have {len(FEATURE_ORDER)}")
            st.stop()
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

    # Navigation
    app_mode = st.sidebar.selectbox(
        "Menu",
        ["📊 Dashboard", "🔄 Live Monitoring", "🎫 Maintenance Tickets", "🛠️ Manual Inspection"],
        index=0
    )
    
    # Add user information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    try:
        conn = sqlite3.connect('turbine_maintenance.db')
        open_tickets = pd.read_sql("SELECT COUNT(*) as count FROM tickets WHERE status = 'Open'", conn)['count'][0]
        st.sidebar.metric("Open Tickets", open_tickets)
    except sqlite3.Error as e:
        st.sidebar.error(f"Failed to get ticket count: {str(e)}")
    finally:
        if conn:
            conn.close()

    if app_mode == "📊 Dashboard":
        st.header("System Dashboard")
        
        try:
            # Initialize database connections
            pred_conn = sqlite3.connect('turbine_monitoring.db')
            ticket_conn = sqlite3.connect('turbine_maintenance.db')
            
            # Check if tables exist
            pred_table_check = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'", pred_conn)
            ticket_table_check = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'", ticket_conn)
            
            if pred_table_check.empty or ticket_table_check.empty:
                st.warning("Database tables not initialized. Creating tables now...")
                maint_system.init_databases()
                time.sleep(1)
                st.rerun()
            
            # ========================
            # SYSTEM OVERVIEW SECTION
            # ========================
            st.subheader("System Overview")
            
            # Get prediction statistics with NULL handling
            pred_stats = pd.read_sql('''
                SELECT 
                    COUNT(*) as total_predictions,
                    COALESCE(AVG(failure_probability), 0) as avg_probability,
                    COALESCE(SUM(CASE WHEN failure_probability > 70 THEN 1 ELSE 0 END), 0) as high_risk_count
                FROM predictions
            ''', pred_conn)
            
            # Get ticket statistics with NULL handling
            ticket_stats = pd.read_sql('''
                SELECT 
                    COALESCE(COUNT(*), 0) as total_tickets,
                    COALESCE(SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END), 0) as open_tickets,
                    COALESCE(SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END), 0) as resolved_tickets
                FROM tickets
            ''', ticket_conn)
            
            # Display KPIs with proper NULL handling
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", int(pred_stats['total_predictions'][0]))
            
            avg_prob = pred_stats['avg_probability'][0]
            col2.metric("Avg Failure Probability", 
                    f"{avg_prob:.1f}%" if avg_prob is not None else "0.0%",
                    delta=None if avg_prob is None else f"{avg_prob - 50:.1f}% from baseline")
            
            col3.metric("High Risk Cases", int(pred_stats['high_risk_count'][0]))
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Total Tickets", int(ticket_stats['total_tickets'][0]))
            col5.metric("Open Tickets", int(ticket_stats['open_tickets'][0]))
            col6.metric("Resolved Tickets", int(ticket_stats['resolved_tickets'][0]))
            
            # ========================
            # INTERACTIVE PREDICTION
            # ========================
            
            with st.expander("🖥️ Interactive Prediction", expanded=False):
            # Create input form
                with st.form("prediction_form"):
                    st.write("Enter turbine parameters for prediction:")
                    
                    # Create input fields for all features
                    user_data = {'Turbine_ID': st.selectbox("Turbine ID", ["WTG-01", "WTG-02", "WTG-03"])}
                    
                    cols = st.columns(3)
                    for i, feature in enumerate(FEATURE_ORDER):
                        with cols[i % 3]:
                            low, high = FEATURE_RANGES[feature]
                            user_data[feature] = st.slider(
                                feature.replace('_', ' ').title(),
                                min_value=float(low),
                                max_value=float(high),
                                value=float((low + high) / 2),
                                step=0.1
                            )
                    
                    submitted = st.form_submit_button("Predict Failure Probability")
                
                if submitted:
                    features = pd.DataFrame(user_data, index=[0])
                    
                    st.subheader("Input Features")
                    st.dataframe(features)
                    
                    # Make prediction
                    try:
                        prediction = model.predict_proba(features[FEATURE_ORDER])[:, 1][0]
                        failure_probability = round(prediction * 100, 2)
                        failure_probability = min(max(failure_probability, 0), 100)
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.metric("Failure Probability", f"{failure_probability}%")
                        
                        if prediction > 0.7:
                            st.error("🚨 Warning: High probability of failure detected!")
                        elif prediction > 0.5:
                            st.warning("⚠️ Caution: Moderate probability of failure detected")
                        else:
                            st.success("✅ Normal operation")
                        
                        # Log the prediction
                        maint_system.log_prediction(
                            user_data['Turbine_ID'],
                            failure_probability,
                            features[FEATURE_ORDER].to_dict('records')[0]
                        )
                        
                        # Check for alerts
                        if prediction > 0.7:
                            issue = f"Manual prediction indicates high failure risk ({failure_probability}%)"
                            ticket_id = maint_system.create_ticket(
                                user_data['Turbine_ID'],
                                issue,
                                failure_probability
                            )
                            st.toast(f"Created maintenance ticket #{ticket_id}")
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

            
            # ========================
            # FEATURE ANALYSIS
            # ========================
            st.subheader("🔍 Feature Analysis")
            pred_df = pd.read_sql('''
                SELECT timestamp, turbine_id, failure_probability, 
                    json_extract(features, '$.rotor_speed') as rotor_speed,
                    json_extract(features, '$.generator_speed') as generator_speed,
                    json_extract(features, '$.power_output') as power_output,
                    json_extract(features, '$.gearbox_temp') as gearbox_temp,
                    json_extract(features, '$.generator_temp') as generator_temp,
                    json_extract(features, '$.vibration') as vibration
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT 100
            ''', pred_conn)
            
            if not pred_df.empty:
                # Convert JSON extracted values to numeric
                numeric_cols = ['rotor_speed', 'generator_speed', 'power_output', 
                            'gearbox_temp', 'generator_temp', 'vibration']
                for col in numeric_cols:
                    pred_df[col] = pd.to_numeric(pred_df[col])
                
                feature = st.selectbox("Select feature to analyze", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"{feature.replace('_', ' ').title()} Over Time")
                    fig = px.line(pred_df, x='timestamp', y=feature, color='turbine_id')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader(f"{feature.replace('_', ' ').title()} vs Failure Probability")
                    fig = px.scatter(pred_df, x=feature, y='failure_probability', 
                                    color='turbine_id', trendline="lowess")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature data available")
            
            # ========================
            # ACTIVE ALERTS SECTION
            # ========================
            # st.subheader("🚨 Active Alerts")
            # alert_df = pd.read_sql('''
            #     SELECT id as ticket_id, turbine_id, created_at as timestamp, 
            #         severity as alert_type, issue as alert_message
            #     FROM tickets
            #     WHERE status = 'Open'
            #     ORDER BY created_at DESC
            # ''', ticket_conn)
            
            # if not alert_df.empty:
            #     for _, row in alert_df.iterrows():
            #         with st.expander(f"{row['timestamp']} - {row['turbine_id']}: {row['alert_type']}"):
            #             st.warning(row['alert_message'])
            #             if st.button(f"Mark as Resolved", key=f"resolve_{row['ticket_id']}"):
            #                 maint_system.resolve_ticket(row['ticket_id'])
            #                 st.rerun()
            # else:
            #     st.success("No active alerts")
            
            # ========================
            # PREDICTION TREND CHART
            # ========================
            st.subheader("Failure Probability Trend (Last 100 Readings)")
            trend_df = pd.read_sql('''
                SELECT timestamp, turbine_id, failure_probability 
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''', pred_conn)
            
            if not trend_df.empty:
                trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'])
                fig = px.line(trend_df, x='timestamp', y='failure_probability', 
                            color='turbine_id', title="Recent Predictions",
                            labels={'failure_probability': 'Failure Probability (%)', 'timestamp': 'Time'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No prediction data available")
                
            # Severity distribution chart
            st.subheader("Ticket Severity Distribution")
            severity_df = pd.read_sql('''
                SELECT severity, COUNT(*) as count
                FROM tickets
                GROUP BY severity
                ORDER BY count DESC
            ''', ticket_conn)
            
            if not severity_df.empty:
                fig = px.pie(severity_df, values='count', names='severity', 
                            title='Ticket Severity Distribution',
                            color='severity',
                            color_discrete_map=maint_system.priority_config)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ticket data available")
                
        except sqlite3.Error as e:
            st.error(f"Database error: {str(e)}")
        finally:
            if 'pred_conn' in locals():
                pred_conn.close()
            if 'ticket_conn' in locals():
                ticket_conn.close()
    # elif app_mode == "🔄 Live Monitoring":
    #     st.header("Real-Time Monitoring")
        
    #     # Simulation controls
    #     col1, col2 = st.columns(2)
    #     turbine_id = col1.selectbox("Select Turbine", ["WTG-01", "WTG-02", "WTG-03"])
    #     num_readings = col2.number_input("Number of Readings", min_value=1, max_value=100, value=10)
        
    #     if st.button("Start Simulation", type="primary"):
    #         placeholder = st.empty()
    #         chart_placeholder = st.empty()
    #         status_placeholder = st.empty()
            
    #         # Initialize chart
    #         fig = go.Figure()
    #         fig.update_layout(
    #             title=f"Live Sensor Data - {turbine_id}",
    #             xaxis_title="Time",
    #             yaxis_title="Value",
    #             height=500
    #         )
            
    #         # Status indicators
    #         status_cols = st.columns(4)
            
    #         for i in range(num_readings):
    #             # Generate complete fake data with ALL features
    #             live_data = {
    #                 'Turbine_ID': turbine_id,
    #                 **{feature: np.random.uniform(low, high) 
    #                    for feature, (low, high) in FEATURE_RANGES.items()}
    #             }
                
    #             # Prepare features in EXACT SAME ORDER as training data
    #             features = np.array([[live_data[feature] for feature in FEATURE_ORDER]])
                
    #             # Make prediction
    #             try:
    #                 probability = model.predict_proba(features)[:, 1][0] * 100
    #                 probability = min(max(probability, 0), 100)  # Clip to 0-100 range
    #             except Exception as e:
    #                 st.error(f"Prediction failed: {str(e)}")
    #                 break
                
    #             # Log prediction
    #             maint_system.log_prediction(
    #                 live_data['Turbine_ID'],
    #                 probability,
    #                 live_data
    #             )
                
    #             # Create ticket if high risk
    #             if probability > 70:
    #                 issue = f"High risk detected (Vibration: {live_data['vibration']:.1f}mm/s, Temp: {live_data['gearbox_temp']:.1f}°C)"
    #                 ticket_id = maint_system.create_ticket(
    #                     live_data['Turbine_ID'],
    #                     issue,
    #                     probability
    #                 )
    #                 st.toast(f"🚨 Created ticket #{ticket_id} for {turbine_id}", icon="⚠️")
                
    #             # Update display
    #             with placeholder.container():
    #                 st.subheader(f"Current Status - {turbine_id}")
    #                 cols = st.columns(3)
    #                 cols[0].metric("Failure Probability", f"{probability:.1f}%", 
    #                              delta=f"{(probability - 50):.1f}% from baseline" if i > 0 else None)
    #                 cols[1].metric("Vibration", f"{live_data['vibration']:.1f} mm/s",
    #                               delta=f"{(live_data['vibration'] - 3):.1f} mm/s from normal" if i > 0 else None)
    #                 cols[2].metric("Gearbox Temp", f"{live_data['gearbox_temp']:.1f}°C",
    #                               delta=f"{(live_data['gearbox_temp'] - 75):.1f}°C from normal" if i > 0 else None)
                    
    #                 with st.expander("View All Sensor Data"):
    #                     st.json({k: v for k, v in live_data.items() if k != 'Turbine_ID'})
                
    #             # Update status indicators
    #             with status_placeholder.container():
    #                 status_cols = st.columns(4)
    #                 status_cols[0].metric("Power Output", f"{live_data['power_output']:.1f} kW")
    #                 status_cols[1].metric("Wind Speed", f"{live_data['wind_speed']:.1f} m/s")
    #                 status_cols[2].metric("Oil Pressure", f"{live_data['oil_pressure']:.1f} psi")
    #                 status_cols[3].metric("Time Since Maintenance", f"{live_data['time_since_maintenance']:.0f} days")
                
    #             # Update chart
    #             fig.add_trace(go.Scatter(
    #                 x=[datetime.now()],
    #                 y=[probability],
    #                 mode='lines+markers',
    #                 name='Failure Probability',
    #                 line=dict(color='red' if probability > 70 else 'orange' if probability > 50 else 'green')
    #             ))
    #             chart_placeholder.plotly_chart(fig, use_container_width=True)
                
    #             time.sleep(2)  # Simulate 2-second delay

    elif app_mode == "🔄 Live Monitoring":
        st.header("Real-Time Monitoring")
        
        # Remove the turbine selection dropdown and use automatic numbering
        num_turbines = 3  # Set this to however many turbines you want to monitor
        turbine_ids = [f"WTG-{i+1:02d}" for i in range(num_turbines)]  # Generates WTG-01, WTG-02, etc.
        
        # Keep only the number of readings input
        num_readings = st.number_input("Number of Readings", min_value=1, max_value=100, value=10)
        
        if st.button("Start Simulation", type="primary"):
            placeholder = st.empty()
            chart_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Initialize chart for all turbines
            fig = go.Figure()
            fig.update_layout(
                title="Live Sensor Data - All Turbines",
                xaxis_title="Time",
                yaxis_title="Failure Probability (%)",
                height=500
            )
            
            # Add a trace for each turbine
            for turbine_id in turbine_ids:
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines+markers',
                    name=turbine_id,
                    line=dict(width=2)
                ))
            
            # Initialize data storage
            all_data = {turbine_id: {'timestamps': [], 'probabilities': []} for turbine_id in turbine_ids}
            
            for i in range(num_readings):
                # Create container for turbine status displays
                status_containers = {turbine_id: st.container() for turbine_id in turbine_ids}
                
                for turbine_id in turbine_ids:
                    # Generate complete fake data with ALL features
                    live_data = {
                        'Turbine_ID': turbine_id,
                        **{feature: np.random.uniform(low, high) 
                        for feature, (low, high) in FEATURE_RANGES.items()}
                    }
                    
                    # Prepare features for prediction
                    features = np.array([[live_data[feature] for feature in FEATURE_ORDER]])
                    
                    # Make prediction
                    try:
                        probability = model.predict_proba(features)[:, 1][0] * 100
                        probability = min(max(probability, 0), 100)  # Clip to 0-100 range
                    except Exception as e:
                        st.error(f"Prediction failed for {turbine_id}: {str(e)}")
                        continue
                    
                    # Log prediction
                    maint_system.log_prediction(
                        turbine_id,
                        probability,
                        live_data
                    )
                    
                    # Create ticket if high risk
                    if probability > 70:
                        issue = f"High risk detected (Vibration: {live_data['vibration']:.1f}mm/s, Temp: {live_data['gearbox_temp']:.1f}°C)"
                        ticket_id = maint_system.create_ticket(
                            turbine_id,
                            issue,
                            probability
                        )
                        st.toast(f"🚨 Created ticket #{ticket_id} for {turbine_id}", icon="⚠️")
                    
                    # Update data storage
                    timestamp = datetime.now()
                    all_data[turbine_id]['timestamps'].append(timestamp)
                    all_data[turbine_id]['probabilities'].append(probability)
                    
                    # Update turbine status display
                    with status_containers[turbine_id]:
                        st.subheader(f"Turbine {turbine_id} Status")
                        cols = st.columns(3)
                        cols[0].metric("Failure Probability", f"{probability:.1f}%", 
                                    delta=f"{(probability - 50):.1f}% from baseline" if i > 0 else None)
                        cols[1].metric("Vibration", f"{live_data['vibration']:.1f} mm/s",
                                    delta=f"{(live_data['vibration'] - 3):.1f} mm/s from normal" if i > 0 else None)
                        cols[2].metric("Gearbox Temp", f"{live_data['gearbox_temp']:.1f}°C",
                                    delta=f"{(live_data['gearbox_temp'] - 75):.1f}°C from normal" if i > 0 else None)
                        
                        with st.expander("View All Sensor Data"):
                            st.json({k: v for k, v in live_data.items() if k != 'Turbine_ID'})
                    
                    # Update the chart trace for this turbine
                    fig.update_traces(
                        x=all_data[turbine_id]['timestamps'],
                        y=all_data[turbine_id]['probabilities'],
                        selector={'name': turbine_id}
                    )
                
                # Update the chart display
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(2)  # Simulate 2-second delay

    elif app_mode == "🎫 Maintenance Tickets":
        st.header("Maintenance Tickets")
        
        # Add ticket filtering options
        col1, col2 = st.columns(2)
        severity_filter = col1.multiselect(
            "Filter by Severity",
            options=list(maint_system.priority_config.keys()),
            default=list(maint_system.priority_config.keys())
        )
        status_filter = col2.selectbox(
            "Filter by Status",
            options=['All', 'Open', 'Resolved'],
            index=1
        )
        
        # Show tickets based on filters
        try:
            conn = sqlite3.connect('turbine_maintenance.db')
            query = "SELECT * FROM tickets WHERE severity IN ({})".format(
                ','.join(['?'] * len(severity_filter))
            )
            params = severity_filter
            
            if status_filter != 'All':
                query += " AND status = ?"
                params.append(status_filter)
                
            query += " ORDER BY created_at DESC"
            
            tickets = pd.read_sql(query, conn, params=params)
            
            if not tickets.empty:
                for _, ticket in tickets.iterrows():
                    with st.expander(f"Ticket #{ticket['id']} - {ticket['turbine_id']} ({ticket['severity']})"):
                        col1, col2 = st.columns(2)
                        col1.write(f"**Issue:** {ticket['issue']}")
                        col1.write(f"**Probability:** {ticket['probability']}%")
                        col2.write(f"**Created:** {pd.to_datetime(ticket['created_at']).strftime('%Y-%m-%d %H:%M')}")
                        if ticket['status'] == 'Resolved':
                            col2.write(f"**Resolved:** {pd.to_datetime(ticket['resolved_at']).strftime('%Y-%m-%d %H:%M')}")
                        
                        if ticket['status'] == 'Open':
                            with st.form(key=f"resolve_{ticket['id']}"):
                                assigned_to = st.text_input("Your Name", key=f"assignee_{ticket['id']}")
                                if st.form_submit_button("Mark Resolved"):
                                    if maint_system.resolve_ticket(ticket['id'], assigned_to):
                                        st.success(f"Ticket #{ticket['id']} marked as resolved!")
                                        time.sleep(1)
                                        st.rerun()
            else:
                st.success("No tickets match the current filters!")
                
            # Show ticket statistics
            st.subheader("Ticket Analytics")
            try:
                ticket_stats = pd.read_sql('''
                    SELECT 
                        severity,
                        COUNT(*) as count,
                        AVG(probability) as avg_probability,
                        AVG(JULIANDAY(resolved_at) - JULIANDAY(created_at)) as avg_resolution_days
                    FROM tickets
                    WHERE status = 'Resolved'
                    GROUP BY severity
                ''', conn)

                # In the Ticket Analytics section, replace the problematic code with:

                if not ticket_stats.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(ticket_stats.style.format({
                            'avg_probability': '{:.1f}%',
                            'avg_resolution_days': '{:.1f} days'
                        }))
                    with col2:
                        # Create a color mapping from severity to color codes
                        color_map = {
                            'Critical': '#d9534f',
                            'High': '#f0ad4e',
                            'Medium': '#5bc0de',
                            'Low': '#5cb85c'
                        }
                        
                        # Map the severity values to colors
                        ticket_stats['color'] = ticket_stats['severity'].map(color_map)
                        
                        fig = px.bar(ticket_stats, 
                                    x='severity', 
                                    y='count', 
                                    color='severity',
                                    color_discrete_map=color_map,
                                    title="Resolved Tickets by Severity")
                        st.plotly_chart(fig, use_container_width=True)
          
            
        
            

                # if not ticket_stats.empty:
                #     col1, col2 = st.columns(2)
                #     with col1:
                #         st.dataframe(ticket_stats.style.format({
                #             'avg_probability': '{:.1f}%',
                #             'avg_resolution_days': '{:.1f} days'
                #         }))
                #     with col2:
                #         fig = px.bar(ticket_stats, x='severity', y='count', 
                #                     color='severity', title="Resolved Tickets by Severity",
                #                     color_discrete_map=maint_system.priority_config)
                #         st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No resolved tickets for analytics")
            except sqlite3.Error as e:
                st.error(f"Failed to load ticket analytics: {str(e)}")
                
        except sqlite3.Error as e:
            st.error(f"Failed to load tickets: {str(e)}")
        finally:
            if conn:
                conn.close()

    elif app_mode == "🛠️ Manual Inspection":
        st.header("Manual Inspection Report")
        
        with st.form("manual_ticket", clear_on_submit=True):
            turbine_id = st.selectbox("Turbine ID", ["WTG-01", "WTG-02", "WTG-03", "WTG-04"])
            
            # Create input fields for all features
            feature_inputs = {}
            cols = st.columns(3)
            for i, feature in enumerate(FEATURE_ORDER):
                with cols[i % 3]:
                    low, high = FEATURE_RANGES[feature]
                    feature_inputs[feature] = st.slider(
                        feature.replace('_', ' ').title(),
                        min_value=float(low),
                        max_value=float(high),
                        value=float((low + high) / 2),
                        step=0.1,
                        help=f"Normal range: {low}-{high}"
                    )
            
            issue = st.text_area("Issue Description", 
                               placeholder="Describe the issue in detail...")
            severity = st.selectbox("Initial Severity Assessment", 
                                  list(maint_system.priority_config.keys()))
            
            if st.form_submit_button("Create Ticket"):
                # Prepare features for prediction
                features = np.array([[feature_inputs[feature] for feature in FEATURE_ORDER]])
                
                # Get probability (or use severity threshold if prediction fails)
                try:
                    probability = model.predict_proba(features)[:, 1][0] * 100
                    probability = min(max(probability, 0), 100)  # Ensure within 0-100 range
                except Exception as e:
                    st.warning(f"Prediction failed, using severity threshold: {str(e)}")
                    probability = maint_system.priority_config[severity]['threshold'] + 1
                
                ticket_id = maint_system.create_ticket(
                    turbine_id,
                    issue,
                    probability
                )
                if ticket_id:
                    st.success(f"Ticket #{ticket_id} created successfully!")
                    st.balloons()
                else:
                    st.error("Failed to create ticket")

if __name__ == "__main__":
    main()