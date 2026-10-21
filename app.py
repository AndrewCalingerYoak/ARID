import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
import tempfile
import os

# Page configuration
st.set_page_config(page_title="ARID - Bat Classifier", page_icon="🦇", layout="wide")

# --- CUSTOM CSS TO WIDEN THE SIDEBAR ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 450px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR: AUTHOR INFO, COMMENTARY & LOGOS ---
with st.sidebar:
    st.image("bathead.png", use_container_width=True)
    st.title("About ARID")
    st.markdown("""
    **Acoustic Rabies Identifier (ARID)** is a bioacoustic diagnostic pipeline designed to classify suspect rabid bat vocalizations.
    """)
    
    st.divider()
    
    st.subheader("Principal Investigator")
    st.image("me.jpg", width=150, caption="Andrew Calinger-Yoak")
    st.markdown("""
    **Andrew Calinger-Yoak**  
    **andrewyoak@gmail.com**  
    *Department of Biology*  
    *Otterbein University*  
    
    Specializing in disease ecology and humane rabies control.
    """)

    st.divider()
    st.subheader("Co-Principal Investigators")
    st.markdown("""
    **Dr. Chloe Goodwin**  
    *University of Georgia*  

    **Colin McCabe**  
    *Collaborator & Co-Investigator*
    """)
    
    st.divider()
    
    st.subheader("Partner Organizations")
    st.markdown("Special thanks to our collaborating institutions and wildlife centers providing data acquisition and ecological support:")
    
    st.markdown("* Otterbein University")
    st.markdown("* The Ohio Wildlife Center, Humane Animal Rescue of Pittsburgh, Nebraska Wildlife Rehab, Humane Indiana Wildlife, Bat World MidCities, Houston Humane Society Wildlife Center, Misfits Rehab, Red Creek Wildlife Center, Bat Education and Rehabilitation of Grainger County, Bat Conservation and Rescue of Virginia, and others")

# --- EMAIL NOTIFICATION HELPER ---
def send_email_alert(audio_bytes, audio_name, prediction_label, confidence, metadata):
    try:
        sender_email = st.secrets["EMAIL_USER"]
        sender_password = st.secrets["EMAIL_PASS"]
        receiver_email = "andrewyoak@gmail.com"

        msg = EmailMessage()
        msg['Subject'] = f"🦇 ARID Submission Alert: {prediction_label.upper()} ({confidence:.1f}%)"
        msg['From'] = sender_email
        msg['To'] = receiver_email

        body = f"""
        A new bat audio file has been processed on ARID.

        --- SUBMISSION METADATA ---
        - Submitter / Facility: {metadata['submitter']}
        - Suspected Species: {metadata['species']}
        - Geographic Location: {metadata['location']}
        - Field Notes: {metadata['notes']}

        --- DIAGNOSTIC RESULT ---
        - Predicted Class: {prediction_label.upper()}
        - Confidence: {confidence:.2f}%
        """
        msg.set_content(body)

        # Attach the audio file
        msg.add_attachment(audio_bytes, maintype='audio', subtype='wav', filename=audio_name)

        # Connect to Gmail SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            
    except Exception as e:
        print(f"Email notification failed: {e}")

# --- AUTHENTICATION & MAIN APP ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 Restricted Access")
        pwd = st.text_input("Enter Password to Access ARID:", type="password")
        if st.button("Login"):
            if pwd == "your_secure_password":  # Customize your password here
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect Password")
        return False
    return True

if check_password():
    st.title("🦇 Acoustic Rabies Identifier (ARID)")
    st.markdown("Upload a bat vocalization audio file and complete all metadata fields below to run automated multi-pulse detection, classification summary, and email reporting.")

    # --- BAT METADATA INPUT FORM ---
    with st.form("bat_metadata_form"):
        st.subheader("Specimen Metadata & Field Details (All Fields Required)")
        col1, col2 = st.columns(2)
        with col1:
            submitter_name = st.text_input("Submitting Organization / Rescuer Name *")
            suspected_species = st.selectbox("Suspected / Visual Species ID", [
                "Select Species...",
                "Big Brown Bat (Eptesicus fuscus)",
                "Little Brown Myotis (Myotis lucifugus)",
                "Red Bat (Lasiurus borealis)",
                "Hoary Bat (Lasiurus cinereus)",
                "Silver-haired Bat (Lasionycteris noctivagans)",
                "Tricolored Bat (Perimyotis subflavus)",
                "Other"
            ])
        with col2:
            location = st.text_input("Geographic Origin (City, State / Facility) *")
            notes = st.text_area("Clinical / Behavioral Notes (e.g., ground-found, neurological signs) *")

        audio_file = st.file_uploader("Upload Bat Call Audio (.wav / .mp3) *", type=["wav", "mp3"])
        submitted_analysis = st.form_submit_button("Analyze Call & Send Report", type="primary")

    if submitted_analysis:
        # --- MANDATORY FORM VALIDATION ---
        missing_fields = []
        if not submitter_name: missing_fields.append("Submitting Organization / Rescuer Name")
        if suspected_species == "Select Species...": missing_fields.append("Suspected / Visual Species ID")
        if not location: missing_fields.append("Geographic Origin")
        if not notes: missing_fields.append("Clinical / Behavioral Notes")
        if audio_file is None: missing_fields.append("Audio File Upload")

        if missing_fields:
            st.error(f"⚠️ Please complete the following required fields before submitting: {', '.join(missing_fields)}")
        else:
            st.audio(audio_file)
            
            try:
                audio_bytes = audio_file.getvalue()
                
                # Save temporarily to process with librosa
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name

                y, sr = librosa.load(tmp_path, sr=None)
                os.unlink(tmp_path) # Clean up temp file
                
                # --- AUTOMATED MULTI-PULSE SEGMENTATION & ANALYSIS ---
                intervals = librosa.effects.split(y, top_db=25)

                st.info(f"📊 **Acoustic Scan Complete:** Detected **{len(intervals)}** distinct vocalization pulses/events in the recording.")

                if len(intervals) > 0:
                    model = joblib.load('bat_model.pkl')
                    scaler = joblib.load('scaler.pkl')
                    le = joblib.load('label_encoder.pkl')

                    pulse_results = []
                    
                    # Evaluate each detected pulse
                    for i, (start, end) in enumerate(intervals):
                        y_pulse = y[start:end]
                        
                        # Skip pulses shorter than 0.05 seconds to avoid noise artifacts
                        if (end - start) / sr < 0.05:
                            continue

                        S = np.abs(librosa.stft(y_pulse, n_fft=2048, hop_length=512))
                        S_db = librosa.amplitude_to_db(S, ref=np.max)
                        mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=40)
                        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

                        X_scaled = scaler.transform(mfccs_mean)
                        pred_idx = model.predict(X_scaled)[0]
                        probs = model.predict_proba(X_scaled)[0]

                        label = le.inverse_transform([pred_idx])[0]
                        conf = probs[pred_idx] * 100

                        pulse_results.append({
                            "Pulse": f"Pulse {i+1}",
                            "Start (s)": f"{start/sr:.2f}s",
                            "End (s)": f"{end/sr:.2f}s",
                            "Prediction": label.upper(),
                            "Confidence": f"{conf:.2f}%"
                        })

                    if pulse_results:
                        st.subheader("Multi-Pulse Analysis Summary Table")
                        st.table(pulse_results)
                        
                        # Select the highest-confidence pulse for the main display and email report
                        best_pulse = max(pulse_results, key=lambda x: float(x['Confidence'].strip('%')))
                        class_label = best_pulse['Prediction'].lower()
                        confidence = float(best_pulse['Confidence'].strip('%'))
                        
                        # Grab the specific audio slice of the best pulse for visualization
                        best_idx = int(best_pulse['Pulse'].replace("Pulse ", "")) - 1
                        p_start, p_end = intervals[best_idx]
                        y_segmented = y[p_start:p_end]
                    else:
                        st.warning("Detected pulses were too brief for feature extraction; analyzing full track.")
                        y_segmented = y
                        S = np.abs(librosa.stft(y_segmented, n_fft=2048, hop_length=512))
                        S_db = librosa.amplitude_to_db(S, ref=np.max)
                        mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=40)
                        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

                        model = joblib.load('bat_model.pkl')
                        scaler = joblib.load('scaler.pkl')
                        le = joblib.load('label_encoder.pkl')

                        X_scaled = scaler.transform(mfccs_mean)
                        prediction_idx = model.predict(X_scaled)[0]
                        probabilities = model.predict_proba(X_scaled)[0]

                        class_label = le.inverse_transform([prediction_idx])[0]
                        confidence = probabilities[prediction_idx] * 100
                else:
                    st.warning("No distinct vocalization pulses isolated; analyzing entire file track.")
                    y_segmented = y
                    S = np.abs(librosa.stft(y_segmented, n_fft=2048, hop_length=512))
                    S_db = librosa.amplitude_to_db(S, ref=np.max)
                    mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=40)
                    mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

                    model = joblib.load('bat_model.pkl')
                    scaler = joblib.load('scaler.pkl')
                    le = joblib.load('label_encoder.pkl')

                    X_scaled = scaler.transform(mfccs_mean)
                    prediction_idx = model.predict(X_scaled)[0]
                    probabilities = model.predict_proba(X_scaled)[0]

                    class_label = le.inverse_transform([prediction_idx])[0]
                    confidence = probabilities[prediction_idx] * 100

                # Display primary consensus results
                st.subheader("Consensus Diagnostic Output")
                st.metric(label="Primary Predicted Class", value=class_label.upper(), delta=f"{confidence:.2f}% Confidence")

                # Spectrogram Visualization
                fig, ax = plt.subplots(figsize=(10, 3.5))
                S_mel = librosa.feature.melspectrogram(y=y_segmented, sr=sr, n_mels=128)
                S_dB_mel = librosa.power_to_db(S_mel, ref=np.max)
                img = librosa.display.specshow(S_dB_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                ax.set(title='Mel-Frequency Spectrogram (Primary Pulse / Segment)')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                st.pyplot(fig)

                # Trigger Email Dispatch
                metadata = {
                    "submitter": submitter_name,
                    "species": suspected_species,
                    "location": location,
                    "notes": notes
                }
                send_email_alert(audio_bytes, audio_file.name, class_label, confidence, metadata)
                st.success("Diagnostic report generated and email notification dispatched successfully!")

            except Exception as e:
                st.error(f"Error processing audio file: {e}")