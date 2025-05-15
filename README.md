# Multi-Modal Accessibility Chat Assistant
Demo : https://youtu.be/J6AXniHfSdM

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.20.0%2B-red)

A versatile chat application with three specialized modes designed to enhance accessibility for users with different needs. This application combines sign language detection, voice recognition, and standard text input to create an inclusive communication experience.

<p align="center">
   <img src="https://github.com/AkramOM606/American-Sign-Language-Detection/assets/162604610/6945d009-8aa7-4bf7-99f8-9743662c5248" width="50%">
</p>

## 🌟 Features

### 🔄 Three Specialized Modes

1. **Standard Mode**
   - Traditional text-based chat interface
   - Clean, intuitive design for general users

2. **Visually Impaired Mode**
   - Voice-controlled interface with automatic audio feedback
   - Continuous listening capability
   - Space key shortcut for quick voice recording
   - Voice commands for navigation and control

3. **Non-Verbal Mode**
   - Real-time American Sign Language (ASL) detection
   - Camera-based hand gesture recognition
   - Automatic space insertion after pauses
   - Visual feedback for detected signs

### 🎯 Accessibility Features

- **Automatic Audio Playback**: All responses are automatically read aloud without requiring user interaction
- **Keyboard Shortcuts**: Press Space to start voice recording at any time
- **Voice Commands**: Navigate and control the application using simple voice commands
- **Continuous Listening**: Hands-free operation with always-on voice recognition
- **Multi-language Support**: Voice recognition and text-to-speech in multiple languages

## 📋 Table of Contents

1. [Features](#-features)
2. [Getting Started](#-getting-started)
3. [Usage](#-usage)
4. [Technical Details](#-technical-details)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Model Training](#model-training)
8. [Contributing](#contributing)
9. [License](#license)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (for sign language detection)
- Microphone (for voice recognition)
- Speakers (for audio output)

### Requirements

- OpenCV
- MediaPipe
- Streamlit
- gTTS (Google Text-to-Speech)
- SpeechRecognition
- NumPy
- Pandas
- TensorFlow

> [!IMPORTANT]
> If you face an error during training from the line converting to the tflite model, use TensorFlow v2.16.1.

### Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/yourusername/multi-modal-accessibility-chat.git
   cd multi-modal-accessibility-chat
   ```

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Application:
   ```bash
   streamlit run v6.py
   ```

## 🔧 Usage

### Standard Mode

1. Type your message in the text input field
2. Press Enter to send
3. View the AI's response in the chat history

### Visually Impaired Mode

1. The application starts in this mode by default with continuous listening enabled
2. Simply speak your question or command
3. Listen to the AI's response which plays automatically
4. **Voice Commands**:
   - "help" - List available commands
   - "clear chat" - Clear conversation history
   - "stop listening" - Pause voice recognition
   - "start listening" - Resume voice recognition
   - "switch to standard mode" - Change to text input mode
   - "switch to non-verbal mode" - Change to sign language mode

5. **Keyboard Shortcuts**:
   - Press **Space** at any time to start voice recording
   - Press **Escape** to stop listening

### Non-Verbal Mode

1. Click "Enable Camera" to activate sign language detection
2. Make hand signs corresponding to ASL letters
3. The application will detect and display the letters
4. A space is automatically added after 2 seconds of no input
5. Click "Submit" when your message is complete

## 📊 Technical Details

### Sign Language Detection

- Uses MediaPipe for hand landmark detection
- Custom-trained model for ASL alphabet recognition
- Real-time processing with OpenCV

### Voice Recognition

- Speech recognition with Google's Speech Recognition API
- Text-to-speech conversion with gTTS
- Automatic audio playback with base64 encoding

### User Interface

- Built with Streamlit for a responsive, interactive experience
- Custom JavaScript for keyboard shortcuts and enhanced accessibility
- Session state management for seamless mode switching

## 🧠 AI Integration

The application uses a language model to generate responses. You'll need to provide your own API key in the sidebar:

1. OpenAI API (default)
   - Requires an API key from [OpenAI](https://platform.openai.com/signup)
   - Enter your API key in the sidebar

## 📚 Model Training

If you wish to train the sign language detection model on your dataset, follow these steps:

### Data Collection

1. **Manual Key Points Data Capturing**

   Activate the manual key point saving mode by pressing "k", which will be indicated as "MODE: Logging Key Point".
   If you press any uppercase letter from "A" to "Z", the key points will be recorded and added to the "model/keypoint_classifier/keypoint.csv" file.

   > [!NOTE]
   > Each time you press the uppercase letter a single entry point is appended to keypoint.csv.

2. **Automated Key Points Data Capturing**

   Activate the automatic key point saving mode by pressing "d", which will change the content of the camera window.

   > [!NOTE]
   > You need to specify the dataset directory in the code.

### Training

Launch the Jupyter Notebook "keypoint_classification.ipynb" and run the cells sequentially from the beginning to the end.
If you wish to alter the number of classes in the training data, adjust the value of "NUM_CLASSES = 26" and make sure to update the labels in the "keypoint_classifier_label.csv" file accordingly.

## 🤝 Contributing

We welcome contributions to enhance this project! Feel free to:

1. Fork the repository.
2. Create a new branch for your improvements.
3. Make your changes and commit them.
4. Open a pull request to propose your contributions.
5. We'll review your pull request and provide feedback promptly.

## 📜 License

This project is licensed under the MIT License: https://opensource.org/licenses/MIT (see LICENSE.md for details).
