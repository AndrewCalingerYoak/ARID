import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="ARID - Bat Classifier", page_icon="🦇", layout="wide")

# Simple Authentication
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
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
                # Load audio array and sample rate directly from uploaded buffer
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