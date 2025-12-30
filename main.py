"""
Document QA System with Speech Interface

This system combines speech recognition, visual document understanding, and text-to-speech
to enable interactive question-answering about document images.

Citations:
----------
Whisper (Speech-to-Text):
    Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).
    Robust speech recognition via large-scale weak supervision.
    arXiv preprint arXiv:2212.04356.

Donut (Document Understanding):
    Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., Hwang, W., Yun, S., 
    Han, D., & Park, S. (2022).
    OCR-free Document Understanding Transformer.
    European Conference on Computer Vision (ECCV).
"""


import whisper
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pyttsx3
import pyaudio
import wave
import numpy as np
import threading
import queue

class DocumentQASystem:
    def __init__(self):
        print("Loading models...")
        
        # Load Whisper for speech-to-text
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        # Load Donut for DocVQA
        print("Loading Donut DocVQA model...")
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Audio recording parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        print("All models loaded successfully!")
    
    def record_audio(self, duration=5, filename="question.wav"):
        """Record audio from microphone"""
        print(f"\nRecording for {duration} seconds...")
        
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                       channels=self.CHANNELS,
                       rate=self.RATE,
                       input=True,
                       frames_per_buffer=self.CHUNK)
        
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        print("Recording finished!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save audio file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return filename
    
    def speech_to_text(self, audio_file):
        """Convert speech to text using Whisper"""
        print("\nTranscribing audio...")
        result = self.whisper_model.transcribe(audio_file)
        question = result["text"]
        print(f"Question: {question}")
        return question
    
    def answer_document_question(self, image_path, question):
        """Answer question about document using Donut"""
        print("\nProcessing document and generating answer...")
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Generate answer
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids
        
        pixel_values = pixel_values.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # Generate output
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        # Decode answer
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = sequence.split("<s_answer>")[1].split("</s_answer>")[0].strip()
        
        return sequence
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        print(f"\nAnswer: {text}")
        print("\nSpeaking answer...")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def run_interactive_session(self, document_path):
        """Run interactive Q&A session"""
        print(f"\n{'='*60}")
        print("DOCUMENT QA SYSTEM WITH SPEECH INTERFACE")
        print(f"{'='*60}")
        print(f"\nLoaded document: {document_path}")
        
        while True:
            print(f"\n{'-'*60}")
            print("\nOptions:")
            print("1. Record question (voice)")
            print("2. Type question (text)")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                # Voice input
                duration = input("Recording duration in seconds (default 5): ").strip()
                duration = int(duration) if duration else 5
                
                audio_file = self.record_audio(duration=duration)
                question = self.speech_to_text(audio_file)
                
            elif choice == "2":
                # Text input
                question = input("\nEnter your question: ").strip()
                if not question:
                    print("No question entered!")
                    continue
                    
            elif choice == "3":
                print("\nExiting...")
                break
                
            else:
                print("Invalid option!")
                continue
            
            # Get answer from document
            answer = self.answer_document_question(document_path, question)
            
            # Speak the answer
            self.text_to_speech(answer)

def main():
    # Example usage
    print("Initializing Document QA System...")
    qa_system = DocumentQASystem()
    
    # Specify your document image path
    document_path = input("\nEnter the path to your document image: ").strip()
    
    # Run interactive session
    qa_system.run_interactive_session(document_path)

if __name__ == "__main__":
    main()