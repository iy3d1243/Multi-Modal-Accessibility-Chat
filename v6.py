import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import os
import speech_recognition as sr
import copy
import csv
import mediapipe as mp
import time
import itertools
import requests
import json
import base64
from groq_api import GroqAPI, AVAILABLE_MODELS
from pathlib import Path

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# Helper functions for sign language detection
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text

    cv2.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image

def draw_bounding_rect(image, brect):
    # Outer rectangle
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image

# Create an instance of the Groq API
groq_api = GroqAPI()

# Sidebar for API settings
with st.sidebar:
    st.title("API Settings")

    # Model selection
    st.subheader("Model Selection")
    model_options = {model["name"]: model["id"] for model in AVAILABLE_MODELS}
    model_descriptions = {model["name"]: model["description"] for model in AVAILABLE_MODELS}

    selected_model_name = st.selectbox(
        "Select a model:",
        options=list(model_options.keys()),
        index=0
    )

    selected_model_id = model_options[selected_model_name]
    groq_api.set_model(selected_model_id)

    st.caption(model_descriptions[selected_model_name])

    # API key input (with default already set)
    st.subheader("API Key")
    st.markdown("""
    A default API key is already provided, but you can use your own for higher rate limits.
    """)

    api_key = st.text_input("Enter your Groq API key (optional):", type="password")
    if api_key:
        groq_api.set_api_key(api_key)
        st.success("API key set successfully!")

    st.markdown("""
    ### How to get a Groq API key:
    1. Go to [Groq Cloud](https://console.groq.com/signup)
    2. Sign up for a free account
    3. Navigate to API Keys section
    4. Create a new API key
    5. Copy and paste it here (optional)
    """)

    st.info("Note: This application uses Groq's ultra-fast LLM API for superior response times.")

# Application title
st.title("Chat Application with AI Assistant")

# Initialize session state for mode selection
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "visually_impaired"  # Start in voice mode by default
    st.session_state.first_run = True
    st.session_state.accessibility_mode = True  # Enable accessibility mode by default

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a container for audio playback at the bottom of the page
audio_container = st.container()
with audio_container:
    st.session_state.audio_player = st.empty()

# Counter for unique audio IDs
if "audio_counter" not in st.session_state:
    st.session_state.audio_counter = 0

# Function to handle keyboard shortcuts
def handle_keyboard_shortcuts():
    # Add JavaScript for keyboard shortcuts
    js_code = """
    <script>
    document.addEventListener('keydown', function(e) {
        // Prevent default space bar behavior (scrolling)
        if (e.key === ' ' && !e.target.matches('input, textarea')) {
            e.preventDefault();
        }

        if (e.key === 'v' || e.key === 'V') {
            // Switch to voice mode
            const voiceButton = document.querySelector('button[kind="secondary"]:has-text("Visually Impaired Mode")');
            if (voiceButton) voiceButton.click();
        } else if (e.key === 's' || e.key === 'S') {
            // Switch to standard mode
            const standardButton = document.querySelector('button[kind="secondary"]:has-text("Standard Mode")');
            if (standardButton) standardButton.click();
        } else if (e.key === 'n' || e.key === 'N') {
            // Switch to non-verbal mode
            const nonVerbalButton = document.querySelector('button[kind="secondary"]:has-text("Non-Verbal Mode")');
            if (nonVerbalButton) nonVerbalButton.click();
        } else if (e.key === ' ' && !e.target.matches('input, textarea')) {
            // Activate voice input (Space key)
            // First try to find the "Respond by Voice" button
            const voiceButton = document.querySelector('button:has-text("Respond by Voice")');
            if (voiceButton) {
                voiceButton.click();
            } else {
                // If in continuous mode, try to find the space trigger button
                const spaceTriggerButton = document.getElementById('space_trigger_button');
                if (spaceTriggerButton) {
                    spaceTriggerButton.click();
                }
            }
        } else if (e.key === 'Escape') {
            // Stop listening (Esc key)
            const stopButton = document.querySelector('button:has-text("Stop Listening")') ||
                              document.querySelector('button:has-text("Pause Listening")');
            if (stopButton) stopButton.click();
        }
    });
    </script>
    """
    st.components.v1.html(js_code, height=0)

# Function to get base64 encoded audio
def get_base64_audio(file_path):
    """Convert audio file to base64 encoded string"""
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return audio_b64

# Helper function to play audio automatically using JavaScript
def play_audio_in_app(text, lang='en'):
    """Generate and play audio automatically without requiring user interaction"""
    # Generate audio file with unique name to prevent caching issues
    st.session_state.audio_counter += 1
    audio_file_path = f"temp_audio_{st.session_state.audio_counter}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(audio_file_path)

    # Get base64 encoded audio
    audio_b64 = get_base64_audio(audio_file_path)

    # Create a unique ID for this audio element
    audio_id = f"auto_audio_{st.session_state.audio_counter}"

    # Create HTML with JavaScript that auto-plays the audio
    # Using data URI with base64 encoded audio to avoid file access issues
    audio_html = f"""
    <audio id="{audio_id}" autoplay="true">
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    <script>
        // Create and play audio element with autoplay
        var audioElement = document.getElementById("{audio_id}");

        // Force play (needed for some browsers)
        var playPromise = audioElement.play();

        if (playPromise !== undefined) {{
            playPromise.then(_ => {{
                console.log("Audio playback started successfully");
            }})
            .catch(error => {{
                console.log("Audio playback was prevented: ", error);
                // Try again with user interaction simulation
                document.addEventListener('click', function() {{
                    audioElement.play();
                }}, {{ once: true }});
            }});
        }}
    </script>
    """

    # Display the HTML with auto-playing audio
    st.session_state.audio_player.empty()
    st.session_state.audio_player.markdown(audio_html, unsafe_allow_html=True)

    # Clean up old audio files to prevent clutter
    try:
        if st.session_state.audio_counter > 5:  # Keep only the 5 most recent files
            for i in range(1, st.session_state.audio_counter - 5):
                old_file = f"temp_audio_{i}.mp3"
                if os.path.exists(old_file):
                    os.remove(old_file)
    except Exception:
        pass  # Ignore errors in cleanup

# Mode selection buttons in a horizontal layout
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Standard Mode", use_container_width=True):
        st.session_state.current_mode = "standard"
with col2:
    if st.button("Visually Impaired Mode", use_container_width=True):
        st.session_state.current_mode = "visually_impaired"
with col3:
    if st.button("Non-Verbal Mode", use_container_width=True):
        st.session_state.current_mode = "non_verbal"

# Display current mode
st.subheader(f"Current Mode: {st.session_state.current_mode.replace('_', ' ').title()}")

# Add keyboard shortcuts
handle_keyboard_shortcuts()

# Display chat history (common across all modes)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Standard Mode ---
if st.session_state.current_mode == "standard":
    st.write("This feature allows you to chat with an AI assistant.")

    if prompt := st.chat_input("Enter your message:", key="chat_input_normal"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Generate response using Groq API
                response_text = groq_api.generate_response(prompt)

                # Display the response
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Visually Impaired Mode ---
elif st.session_state.current_mode == "visually_impaired":
    st.write("This feature allows voice interaction with the chatbot.")

    # Initialize voice mode settings in session state
    if "voice_mode_initialized" not in st.session_state:
        st.session_state.voice_mode_initialized = True
        st.session_state.continuous_listening = True  # Enable continuous listening by default
        st.session_state.voice_language = "en-US"
        st.session_state.listening_active = True  # Start listening immediately
        st.session_state.voice_commands = {
            "clear chat": "clear the chat history",
            "stop listening": "pause voice recognition",
            "start listening": "resume voice recognition",
            "help": "show available voice commands",
            "switch to standard mode": "switch to standard text mode",
            "switch to voice mode": "switch to voice mode",
            "switch to non-verbal mode": "switch to sign language mode"
        }

        # Automatically play welcome message for first-time users or when accessibility mode is enabled
        if st.session_state.first_run or st.session_state.accessibility_mode:
            welcome_text = "Welcome to the voice assistant. Voice mode is now active and listening. You can speak commands or questions at any time. Say help to hear available commands."
            play_audio_in_app(welcome_text, lang='en')
            # Add a slight delay to ensure the welcome message is heard
            time.sleep(1)
            # Also announce that continuous listening is active
            listening_text = "Continuous listening is now active. You can speak at any time."
            play_audio_in_app(listening_text, lang='en')
            st.session_state.first_run = False

    # Add information note about voice mode features
    st.info("🎙️ **Voice Mode Instructions:**\n\n"
            "1. Press **SPACE** key anytime to start voice recording (works in any mode)\n"
            "2. Toggle 'Continuous Listening' on/off to enable/disable voice recognition\n"
            "3. Say 'stop listening' to pause voice recognition\n"
            "4. Say 'start listening' to resume voice recognition\n"
            "5. Say 'help' to see all available voice commands\n"
            "6. Say 'clear chat' to clear the conversation\n"
            "7. Click 'Respond by Voice' for single response mode")

    # Voice mode settings
    col1, col2 = st.columns(2)
    with col1:
        # Always show the toggle, but set it to on by default for first run
        continuous_mode = st.toggle("🔄 Continuous Listening", value=st.session_state.continuous_listening)
        if continuous_mode != st.session_state.continuous_listening:
            st.session_state.continuous_listening = continuous_mode
            if continuous_mode:
                st.session_state.listening_active = True
                # Announce that listening is now active
                play_audio_in_app("Continuous listening enabled. You can speak at any time.", lang='en')
                st.rerun()
            else:
                st.session_state.listening_active = False
                # Announce that listening is now disabled
                play_audio_in_app("Continuous listening disabled.", lang='en')
                st.rerun()

    with col2:
        language_options = {
            "English (US)": "en-US",
            "English (UK)": "en-GB",
            "French": "fr-FR",
            "Spanish": "es-ES",
            "German": "de-DE"
        }
        selected_language = st.selectbox(
            "🌐 Language",
            options=list(language_options.keys()),
            index=list(language_options.values()).index(st.session_state.voice_language)
        )
        st.session_state.voice_language = language_options[selected_language]

    # Voice recording status indicator
    status_placeholder = st.empty()

    # Function to process voice input
    def process_voice_input(audio_data, recognizer):
        try:
            user_text = recognizer.recognize_google(audio_data, language=st.session_state.voice_language)

            # Check for voice commands
            if user_text.lower() == "clear chat":
                st.session_state.messages = []
                status_placeholder.success("💬 Chat history cleared!")
                # Provide audio feedback
                play_audio_in_app("Chat history cleared", lang=st.session_state.voice_language[:2])
                return None

            elif user_text.lower() == "help":
                help_text = "Available voice commands:\n"
                for cmd, desc in st.session_state.voice_commands.items():
                    help_text += f"• '{cmd}': {desc}\n"

                status_placeholder.info(help_text)

                # Provide audio feedback for help commands
                help_audio = "Available commands are: clear chat, stop listening, start listening, help, switch to standard mode, switch to voice mode, and switch to non-verbal mode."
                play_audio_in_app(help_audio, lang=st.session_state.voice_language[:2])
                return None

            elif user_text.lower() == "stop listening" and st.session_state.continuous_listening:
                # Just pause listening, don't turn off continuous mode
                st.session_state.listening_active = False
                status_placeholder.warning("🛑 Listening paused")

                # Provide audio feedback
                play_audio_in_app("Listening paused. Say start listening to resume.", lang=st.session_state.voice_language[:2])
                st.rerun()
                return None

            elif user_text.lower() == "start listening" and st.session_state.continuous_listening and not st.session_state.listening_active:
                # Resume listening
                st.session_state.listening_active = True
                status_placeholder.info("🎙️ Listening resumed")

                # Provide audio feedback
                play_audio_in_app("Listening resumed. You can speak now.", lang=st.session_state.voice_language[:2])
                st.rerun()
                return None

            elif user_text.lower() == "switch to standard mode":
                st.session_state.current_mode = "standard"

                # Provide audio feedback
                play_audio_in_app("Switching to standard text mode", lang=st.session_state.voice_language[:2])
                st.rerun()
                return None

            elif user_text.lower() == "switch to voice mode":
                # Already in voice mode, just confirm
                play_audio_in_app("Already in voice mode", lang=st.session_state.voice_language[:2])
                return None

            elif user_text.lower() == "switch to non-verbal mode":
                st.session_state.current_mode = "non_verbal"

                # Provide audio feedback
                play_audio_in_app("Switching to sign language mode", lang=st.session_state.voice_language[:2])
                st.rerun()
                return None

            # Regular message processing
            status_placeholder.write("📝 You said: " + user_text)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_text})

            try:
                # Generate response using Groq API
                response_text = groq_api.generate_response(user_text)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Convert text to speech and play directly in the app
                play_audio_in_app(response_text, lang=st.session_state.voice_language[:2])

                return response_text

            except Exception as e:
                error_message = f"Error: {str(e)}"
                status_placeholder.error("🚨 Error: " + error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                return None

        except sr.UnknownValueError:
            status_placeholder.warning("😕 Could not understand the audio. Please try again.")
            return None

        except sr.RequestError as e:
            status_placeholder.error(f"🚨 Error with the speech recognition service: {e}")
            return None

    # Hidden button that will be triggered by the space key
    # Use a container with CSS to hide the button but keep it functional
    space_trigger_container = st.container()
    with space_trigger_container:
        # Apply CSS to hide the button
        st.markdown("""
        <style>
        div[data-testid="stButton"] #space_trigger_button {
            visibility: hidden;
            height: 0px;
            position: absolute;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create the button that will be hidden but still functional
        if st.button("Space Trigger", key="space_trigger_button", help="This button is triggered when you press the space key"):
            # This will be triggered when the space key is pressed
            status_placeholder.info("🎙️ Space key pressed! Listening... Speak now")
            play_audio_in_app("Space key pressed. Listening now. Speak your message.", lang=st.session_state.voice_language[:2])

            # Initialize recognizer
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)

            # Process the recorded audio
            response = process_voice_input(audio, recognizer)
            if response:
                status_placeholder.success("✅ Response generated")

    # Add a note about the space key shortcut
    st.info("💡 **Tip:** Press the **SPACE** key at any time to start voice recording")

    # Single voice response button
    if not st.session_state.continuous_listening:
        if st.button("🎤 Respond by Voice", use_container_width=True):
            status_placeholder.info("🎙️ Listening... Speak now")

            # Initialize recognizer
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)

            # Process the recorded audio
            response = process_voice_input(audio, recognizer)
            if response:
                status_placeholder.success("✅ Response generated")

    # Continuous listening mode
    else:
        # Only show listening UI if continuous listening is enabled
        if st.session_state.continuous_listening:
            # Show stop button when actively listening
            if st.session_state.listening_active:
                if st.button("🛑 Pause Listening", use_container_width=True):
                    st.session_state.listening_active = False
                    status_placeholder.info("Listening paused")
                    play_audio_in_app("Listening paused. Click Start Listening to resume.", lang='en')
                    st.rerun()

                # Show active listening status
                status_placeholder.info("🔄 Continuous listening mode active... Speak anytime")
            # Show start button when listening is paused
            else:
                if st.button("🎙️ Start Listening", use_container_width=True):
                    st.session_state.listening_active = True
                    play_audio_in_app("Listening started. You can speak now.", lang='en')
                    st.rerun()

                status_placeholder.warning("⏸️ Listening is paused. Click Start Listening to resume.")

            # Initialize recognizer for continuous mode
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)

                try:
                    # Short timeout to allow the UI to remain responsive
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

                    # Process the recorded audio
                    response = process_voice_input(audio, recognizer)
                    if response:
                        status_placeholder.success("✅ Response generated")
                        # Keep listening
                        st.rerun()

                except (sr.WaitTimeoutError, Exception) as e:
                    # Just continue listening
                    st.rerun()

# --- Non-Verbal Mode ---
elif st.session_state.current_mode == "non_verbal":
    st.write("This feature allows communication through sign language.")

    # Initialize sign language detection components
    if "sign_language_initialized" not in st.session_state:
        st.session_state.sign_language_initialized = False
        st.session_state.detected_text = ""
        st.session_state.last_detection_time = time.time()
        st.session_state.last_input_time = time.time()  # Track time since last input for auto-space

    # Add information note about auto-space feature
    st.info("📝 **Sign Language Mode Instructions:**\n\n"
            "1. A space will be automatically added after 2 seconds of no input\n"
            "2. Make hand signs to detect letters\n"
            "3. Click 'Submit' when your message is complete")

    # Create a container for the detected text display
    text_display = st.empty()

    # Display the current detected text
    text_display.text(f"Detected Text: {st.session_state.detected_text}")

    # Button to submit detected text
    col1, col2 = st.columns([3, 1])
    with col1:
        # Just for visual consistency, show a disabled text input
        st.text_input("", value=st.session_state.detected_text, key="detected_text_input", disabled=True)
    with col2:
        if st.button("Submit", use_container_width=True):
            if st.session_state.detected_text:
                # Add detected text to chat history
                st.session_state.messages.append({"role": "user", "content": st.session_state.detected_text})

                try:
                    # Generate response using Groq API
                    response_text = groq_api.generate_response(st.session_state.detected_text)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    # Clear detected text
                    st.session_state.detected_text = ""
                    st.rerun()
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.rerun()

    # Live video recognition
    st.subheader("Live Video Recognition")
    video_active = st.checkbox("Enable Camera")
    stframe = st.empty()

    if video_active:
        # Initialize MediaPipe hands
        if not st.session_state.sign_language_initialized:
            mp_hands = mp.solutions.hands
            st.session_state.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )

            # Initialize keypoint classifier
            st.session_state.keypoint_classifier = KeyPointClassifier()

            # Read labels
            with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
                keypoint_classifier_labels = csv.reader(f)
                st.session_state.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

            st.session_state.sign_language_initialized = True

        # Start video capture
        cap = cv2.VideoCapture(0)

        while video_active:
            ret, image = cap.read()
            if not ret:
                st.write("Error: Unable to capture image")
                break

            # Mirror display
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)

            # Detection implementation
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = st.session_state.hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)

                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    # Hand sign classification
                    hand_sign_id = st.session_state.keypoint_classifier(pre_processed_landmark_list)

                    # Drawing part
                    debug_image = draw_bounding_rect(debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        st.session_state.keypoint_classifier_labels[hand_sign_id],
                    )

                    # Detect letter
                    detected_letter = st.session_state.keypoint_classifier_labels[hand_sign_id]
                    current_time = time.time()

                    if detected_letter != "None":
                        if current_time - st.session_state.last_detection_time >= 1.5:
                            st.session_state.detected_text += detected_letter
                            st.session_state.last_detection_time = current_time
                            st.session_state.last_input_time = current_time  # Reset auto-space timer when new letter is added
                            # Update the text display without refreshing the page
                            # We'll handle this in the main loop

            # Check if we need to add a space (if 2 seconds have passed since last input)
            current_time = time.time()
            if (st.session_state.detected_text and  # Only add space if there's text
                current_time - st.session_state.last_input_time >= 2.0 and  # 2 seconds passed
                not st.session_state.detected_text.endswith(" ")):  # Don't add multiple spaces
                st.session_state.detected_text += " "
                st.session_state.last_input_time = current_time  # Reset timer

            # Update the text display with the current detected text
            text_display.text(f"Detected Text: {st.session_state.detected_text}")

            # Display the frame
            stframe.image(debug_image, channels="RGB")

        cap.release()

# Helper functions for sign language detection
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text

    cv2.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image

def draw_bounding_rect(image, brect):
    # Outer rectangle
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image
