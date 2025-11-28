import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
from datetime import datetime
import time
import speech_recognition as sr
import pyttsx3
import tempfile

# Configure page
st.set_page_config(
    page_title="AI Plant Disease Assistant",
    page_icon="🌱",
    layout="wide"
)

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyCVY9DmgLuqVGRiA-EeKVAwAyymdVUoBxQ"
genai.configure(api_key=GEMINI_API_KEY)

# Load disease prediction model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

# Load class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'interaction_mode' not in st.session_state:
    st.session_state.interaction_mode = "Image Analysis"
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0
if 'voice_mode' not in st.session_state:
    st.session_state.voice_mode = False
if 'pending_audio_response' not in st.session_state:
    st.session_state.pending_audio_response = None

# ==================== HELPER FUNCTIONS ====================

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image for CNN model"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    """Predict disease class from image"""
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence

def get_conversation_context(window_size=3):
    """
    Get recent conversation history for short-term memory
    Returns last N exchanges (user + assistant pairs)
    """
    if not st.session_state.chat_history:
        return ""
    
    # Get last N*2 messages (N exchanges = N user + N assistant messages)
    recent_messages = st.session_state.chat_history[-(window_size * 2):]
    
    # Format as conversation context
    context_parts = []
    for msg in recent_messages:
        role = "User" if msg['role'] == 'user' else "Assistant"
        # Remove voice emoji if present
        content = msg['content'].replace('🎤 ', '').replace('🔊 ', '')
        context_parts.append(f"{role}: {content}")
    
    context = "\n".join(context_parts)
    
    # Add memory info
    total_messages = len(st.session_state.chat_history)
    forgotten_count = max(0, total_messages - (window_size * 2))
    
    memory_note = f"\n\n[Memory: Remembering last {len(recent_messages)} messages"
    if forgotten_count > 0:
        memory_note += f", {forgotten_count} older messages forgotten"
    memory_note += "]"
    
    return context + memory_note

def throttle_request():
    """Ensure we don't exceed rate limits"""
    current_time = time.time()
    
    if current_time - st.session_state.last_request_time > 60:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    
    if st.session_state.request_count >= 14:
        wait_time = 60 - (current_time - st.session_state.last_request_time)
        if wait_time > 0:
            time.sleep(wait_time)
            st.session_state.request_count = 0
            st.session_state.last_request_time = time.time()
    
    st.session_state.request_count += 1

def get_gemini_response(prompt, image=None, use_memory=True, max_retries=3):
    """Get response from Gemini with short-term memory context"""
    throttle_request()
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Build prompt with conversation memory
        full_prompt = prompt
        if use_memory:
            conv_context = get_conversation_context(window_size=3)
            if conv_context:
                full_prompt = f"Previous conversation:\n{conv_context}\n\nCurrent question: {prompt}\n\nRespond naturally considering the recent conversation context."
        
        # Add disease detection context if available
        if st.session_state.last_prediction and "detected disease" not in prompt.lower():
            full_prompt += f"\n\n(Note: Last detected plant disease was: {st.session_state.last_prediction})"
        
        for attempt in range(max_retries):
            try:
                if image:
                    response = model.generate_content([full_prompt, image])
                else:
                    response = model.generate_content(full_prompt)
                
                return response.text
            
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 10
                        time.sleep(wait_time)
                        continue
                    else:
                        return "⚠️ Rate limit exceeded. Please wait a moment and try again."
                else:
                    raise e
        
        return "Error: Max retries reached"
                    
    except Exception as e:
        return f"Error: {str(e)}"

def get_detailed_analysis(disease_name, confidence, image):
    """Get detailed analysis from Gemini"""
    prompt = f"""You are an agricultural expert. Disease detected: {disease_name} ({confidence*100:.1f}% confidence).
    
    Provide concisely:
    1. Key symptoms
    2. Main causes  
    3. Treatment steps
    4. Prevention tips
    
    Keep response under 150 words."""
    
    return get_gemini_response(prompt, image=image, use_memory=False)

# ==================== VOICE FUNCTIONS ====================

def transcribe_audio_opensource(audio_bytes):
    """Transcribe audio using Google Speech Recognition (FREE)"""
    try:
        recognizer = sr.Recognizer()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes.getbuffer())
            tmp_filename = tmp_file.name
        
        with sr.AudioFile(tmp_filename) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        try:
            os.unlink(tmp_filename)
        except:
            pass
        
        return text
    
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None
    except Exception as e:
        return None

def text_to_speech_opensource(text):
    """Convert text to speech using pyttsx3 (OFFLINE & FREE)"""
    try:
        engine = pyttsx3.init()
        
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_filename = tmp_file.name
        
        engine.save_to_file(text, tmp_filename)
        engine.runAndWait()
        
        with open(tmp_filename, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        try:
            os.unlink(tmp_filename)
        except:
            pass
        
        return audio_bytes
    
    except Exception as e:
        return None

# ==================== UI DESIGN ====================

st.title("🌱 AI Plant Disease Assistant")
st.markdown("*Powered by CNN + Gemini AI | Voice & Chat Enabled*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    interaction_mode = st.radio(
        "Interaction Mode",
        ["Image Analysis", "Chat Assistant"],
        key="mode_selector"
    )
    st.session_state.interaction_mode = interaction_mode
    
    st.divider()
    
    # Memory indicator
    if st.session_state.chat_history:
        total_msgs = len(st.session_state.chat_history)
        remembered = min(total_msgs, 6)  # 3 exchanges = 6 messages
        forgotten = max(0, total_msgs - 6)
        
        st.subheader("🧠 Memory Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total_msgs)
            st.metric("Remembering", remembered)
        with col2:
            if forgotten > 0:
                st.metric("Forgotten", forgotten, delta=None, delta_color="off")
        
        st.caption("💡 Short-term memory: Last 3 exchanges")
        st.divider()
    
    st.subheader("📊 Features")
    st.markdown("""
    - 🖼️ **Image Analysis**: CNN disease detection
    - 💬 **Text Chat**: Type your questions
    - 🎤 **Voice Chat**: Speak naturally
    - 🧠 **Memory**: Remembers last 3 exchanges
    - 🔊 **Voice Response**: AI speaks back
    """)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pending_audio_response = None
        st.rerun()

# ==================== IMAGE ANALYSIS MODE ====================

if interaction_mode == "Image Analysis":
    st.header("🖼️ Plant Disease Image Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "Upload Plant Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the affected plant"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button('🔍 Analyze Disease', type="primary", use_container_width=True):
                with st.spinner('Analyzing image...'):
                    prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
                    st.session_state.last_prediction = prediction
                    st.session_state.last_image = image
                    
                    st.session_state.analysis_result = {
                        'disease': prediction,
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
    
    with col2:
        if st.session_state.last_prediction:
            result = st.session_state.analysis_result
            
            st.success(f"**Detected Disease:** {result['disease']}")
            st.metric("Confidence Score", f"{result['confidence']*100:.2f}%")
            
            with st.spinner('Getting AI insights...'):
                detailed_analysis = get_detailed_analysis(
                    result['disease'],
                    result['confidence'],
                    st.session_state.last_image
                )
                
                st.subheader("🤖 AI Analysis")
                st.markdown(detailed_analysis)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"📸 Image Analysis Result:\n\n**Disease:** {result['disease']}\n**Confidence:** {result['confidence']*100:.1f}%\n\n{detailed_analysis}",
                    'timestamp': result['timestamp'],
                    'has_voice': False
                })
                
                st.success("✅ Analysis added to chat history. Switch to Chat Assistant to ask follow-up questions!")

# ==================== UNIFIED CHAT ASSISTANT MODE ====================

elif interaction_mode == "Chat Assistant":
    st.header("💬 Chat Assistant (Text + Voice)")
    
    # Voice mode toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.session_state.voice_mode = st.toggle(
            "🎤 Voice Mode", 
            value=st.session_state.voice_mode,
            help="Enable to send voice messages and receive voice responses"
        )
    with col2:
        if st.session_state.voice_mode:
            st.success("Voice ON 🔊")
        else:
            st.info("Text Mode 💬")
    
    # Display chat history
    chat_container = st.container(height=400)
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
                # Show voice response if available
                if message['role'] == 'assistant' and message.get('has_voice') and message.get('audio_data'):
                    st.audio(message['audio_data'], format='audio/wav')
    
    # Input area - switches between text and voice
    if st.session_state.voice_mode:
        # Voice input
        st.markdown("### 🎤 Voice Input")
        audio_input = st.audio_input("Record your question")
        
        if audio_input:
            st.audio(audio_input, format="audio/wav")
            
            if st.button("🎯 Send Voice Message", type="primary", use_container_width=True):
                # Transcribe
                with st.spinner("🎧 Transcribing..."):
                    transcribed_text = transcribe_audio_opensource(audio_input)
                
                if transcribed_text:
                    st.success(f"📝 Transcribed: *{transcribed_text}*")
                    
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': f"🎤 {transcribed_text}",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'has_voice': False
                    })
                    
                    # Get AI response with memory
                    with st.spinner('🤖 AI is thinking...'):
                        ai_response = get_gemini_response(
                            f"Answer this question in 2-3 sentences: {transcribed_text}",
                            use_memory=True
                        )
                    
                    # Generate voice response
                    with st.spinner('🔊 Generating voice...'):
                        audio_response = text_to_speech_opensource(ai_response)
                    
                    # Add assistant message
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"🔊 {ai_response}",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'has_voice': True,
                        'audio_data': audio_response
                    })
                    
                    st.rerun()
                else:
                    st.error("❌ Could not understand audio. Please try again.")
        
        # Quick tips
        with st.expander("💡 Voice Chat Tips"):
            st.markdown("""
            - Speak clearly for 2-5 seconds
            - Ask one question at a time
            - The AI will respond with voice
            - Previous context is remembered
            
            **Example questions:**
            - "What are the symptoms?"
            - "How should I treat this?"
            - "Tell me more about prevention"
            """)
    
    else:
        # Text input
        user_input = st.chat_input("💬 Type your question about plant diseases...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'has_voice': False
            })
            
            # Get AI response with memory
            with st.spinner('🤖 Thinking...'):
                ai_response = get_gemini_response(
                    f"Answer this question concisely (2-3 sentences): {user_input}",
                    use_memory=True
                )
            
            # Add assistant message
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'has_voice': False
            })
            
            st.rerun()

# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Messages", len(st.session_state.chat_history))
with col2:
    if st.session_state.last_prediction:
        st.metric("Last Detection", st.session_state.last_prediction.split('_')[0])
with col3:
    st.metric("Mode", "Voice 🎤" if st.session_state.voice_mode else "Text 💬")
with col4:
    memory_count = min(len(st.session_state.chat_history), 6)
    st.metric("Memory", f"{memory_count//2}/3 exchanges")
