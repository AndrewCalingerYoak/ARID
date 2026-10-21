import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt



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

# --- SIDEBAR: AUTHOR INFO, COMMENTARY & LOGOS ---
with st.sidebar:
    st.image("bathead.png",  use_container_width=True)
    st.title("About ARID")
    st.markdown("""
    **Bat Acoustic Rabies Identifier (ARID)** is a bioacoustic diagnostic pipeline designed to classify suspect rabid bat vocalizations.
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
    st.subheader("Co Investigators")
    st.markdown("""
    **Dr Chloe Goodwin**
    *University of Georgia*
    """)
    
    
    st.divider()
    
    st.subheader("Partner Organizations")
    st.markdown("Special thanks to our collaborating institutions and wildlife centers providing data acquisition and support:")
    
    # You can swap these with local image filenames once uploaded to GitHub, e.g., st.image("otterbein_logo.png", width=150)
    st.markdown("* Otterbein University")
    st.markdown("* The Ohio Wildlife Center, Humane Animal Rescue of Pittsburgh, Nebraska Wildlife Rehab, Humane Indiana Wildlife, Bat World MidCities, Houston Humane Society Wildlife Center, Misfits Rehab, Red Creek Wildlife Center, Bat Education and Rehabilitation of Grainger County, Bat Conservation and Rescue of Virginia, and others")

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
    st.title("🦇 Bat Acoustic Rabies Identifier (ARID)")
    st.markdown("Upload or record a bat vocalization audio file below to run linear-frequency feature extraction and classification.")

    # Load model artifacts
    model = joblib.load('bat_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')

    audio_file = st.file_uploader("Upload Bat Call Audio (.wav / .mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("Analyze Call", type="primary"):
            try:
                y, sr = librosa.load(audio_file, sr=None)
                
                # Extract linear-frequency MFCCs
                S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=40)
                mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

                # Scale and predict
                X_scaled = scaler.transform(mfccs_mean)
                prediction_idx = model.predict(X_scaled)[0]
                probabilities = model.predict_proba(X_scaled)[0]

                class_label = le.inverse_transform([prediction_idx])[0]
                confidence = probabilities[prediction_idx] * 100

                # Display classification metric
                st.subheader("Diagnostic Output")
                st.metric(label="Predicted Class", value=class_label.upper(), delta=f"{confidence:.2f}% Confidence")

                # Spectrogram Visualization
                fig, ax = plt.subplots(figsize=(10, 3.5))
                S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB_mel = librosa.power_to_db(S_mel, ref=np.max)
                img = librosa.display.specshow(S_dB_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                ax.set(title='Mel-Frequency Spectrogram')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error processing audio file: {e}")