import customtkinter as ctk
import threading
import time
import random
from ui_design import ChatInput, ChatDisplay

class ChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DSA4265 Chatbot")
        self.geometry("750x850")
        
        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self._setup_ui()
        self._send_welcome_message()
    
    def _setup_ui(self):
        """Initialize all UI components"""
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create chat display
        self.chat_display = ChatDisplay(self)
        self.chat_display.grid(
            row=0, column=0, 
            padx=20, pady=(20, 10), 
            sticky="nsew"
        )
        
        # Create input area with proper expansion
        self.input_area = ChatInput(
            self,
            send_callback=self.send_message,
            fg_color="transparent"
        )
        self.input_area.grid(
            row=1, column=0, 
            padx=20, pady=(0, 20), 
            sticky="nsew"
        )
        
        # Make sure input area expands properly
        self.grid_rowconfigure(1, weight=0)
    
    def _send_welcome_message(self):
        """Send initial welcome message"""
        self.chat_display.add_message(
            "🤖 Bot", 
            "Hello! I'm your AI assistant. How can I help you today?", 
            True
        )
    
    def send_message(self, message):
        """Handle sending a message"""
        if not message.strip():
            return
            
        self.chat_display.add_message("👤 You", message, False)
        self.chat_display.show_typing()
        
        # Simulate AI response in background
        threading.Thread(
            target=self._generate_response,
            args=(message,),
            daemon=True
        ).start()
    
    def _generate_response(self, message):
        """Generate bot response"""
        time.sleep(1)  # Simulate processing delay
        
        responses = [
            f"I understand you're asking about: {message}",
            f"Interesting question! Here's what I think about {message}...",
            "I'm a simple chatbot, but I can help with many topics!",
            "Let me check my knowledge base about that...",
            "Great question! Here's what I found."
        ]
        response = random.choice(responses)
        
        self.after(0, self._display_response, response)
    
    def _display_response(self, response):
        """Display the generated response"""
        self.chat_display.hide_typing()
        self.chat_display.add_message("🤖 Bot", response, True)

if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()