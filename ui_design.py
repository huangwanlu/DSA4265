import customtkinter as ctk

class ChatInput(ctk.CTkFrame):
    def __init__(self, master, send_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.send_callback = send_callback
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Text widget configuration
        self.text_widget = ctk.CTkTextbox(
            self,
            wrap="word",
            font=("Arial", 14),
            height=40,  # Initial height (~2 lines)
            activate_scrollbars=True,
        )
        self.text_widget.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Bind events
        self.text_widget.bind("<Key>", self._schedule_update)
        self.text_widget.bind("<KeyRelease>", self._schedule_update)
        self.text_widget.bind("<Return>", self._on_enter_press)
        self.text_widget.bind("<Shift-Return>", lambda e: None)
        
        # Send button
        self.send_button = ctk.CTkButton(
            self,
            text="Send",
            width=70,
            command=self._send_message
        )
        self.send_button.grid(row=0, column=1, sticky="ne")
        
        # Tracking variables
        self._update_pending = False
        self._last_line_count = 1
        self._max_visible_lines = 5
        self._line_height = 20  # Pixels per line
        
    def _schedule_update(self, event=None):
        """Schedule a height update if one isn't already pending"""
        if not self._update_pending:
            self._update_pending = True
            self.after(10, self._update_textbox_height)
    
    def _update_textbox_height(self):
        """Calculate and update textbox height based on content"""
        self._update_pending = False
        
        # Get current content and calculate display lines
        content = self.text_widget.get("1.0", "end-1c")
        
        # Method 1: Count physical lines (including wrapped text)
        self.text_widget.update_idletasks()  # Ensure UI is up-to-date
        line_count = int(self.text_widget.index("end-1c").split('.')[0])
        
        # Method 2: Alternative calculation for wrapped lines
        widget_width = self.text_widget.winfo_width()
        if widget_width > 1:  # Ensure widget is visible
            avg_chars_per_line = widget_width // 8  # Approximate based on font
            if avg_chars_per_line > 0:
                line_count = max(1, len(content) // avg_chars_per_line)
        
        # Use the larger of the two line counts
        line_count = max(line_count, content.count('\n') + 1)
        
        # Only update if line count changed
        if line_count != self._last_line_count:
            self._last_line_count = line_count
            
            # Calculate new height (capped at max visible lines)
            visible_lines = min(line_count, self._max_visible_lines)
            new_height = 40 + (visible_lines - 1) * self._line_height
            
            # Apply height change
            self.text_widget.configure(height=new_height)
            
            # Scroll to show insertion cursor
            self.text_widget.see("insert")
    
    def _on_enter_press(self, event):
        """Handle Enter key (send) vs Shift+Enter (newline)"""
        if not event.state & 0x1:  # If Shift is not pressed
            self._send_message()
            return "break"
        # Schedule update after the newline is inserted
        self._schedule_update()
        return None
    
    def _send_message(self):
        """Send the current message"""
        message = self.text_widget.get("1.0", "end-1c").strip()
        if message:
            self.send_callback(message)
            self.text_widget.delete("1.0", "end")
            self._last_line_count = 1
            self.text_widget.configure(height=40)
    
    def get(self):
        """Get current text content"""
        return self.text_widget.get("1.0", "end-1c")
    
    def focus(self):
        """Focus the input field"""
        self.text_widget.focus()

class ChatDisplay(ctk.CTkTextbox):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            wrap="word",
            font=("Arial", 14), 
            state="disabled",
            scrollbar_button_color="#4FC3F7",
            scrollbar_button_hover_color="#2A9CC8"
        )
        
        # Configure tags
        self.tag_config("bot", foreground="#4FC3F7")
        self.tag_config("user", foreground="#BA68C8")
        self.tag_config("typing", foreground="#A0A0A0")
    
    def add_message(self, sender, message, is_bot):
        self.configure(state="normal")
        tag = "bot" if is_bot else "user"
        self.insert("end", f"{sender}: ", tag)
        self.insert("end", f"{message}\n\n")
        self.configure(state="disabled")
        self.see("end")
    
    def show_typing(self):
        self.configure(state="normal")
        self.insert("end", "⌛ Bot is thinking...\n", "typing")
        self.see("end")
        self.configure(state="disabled")
    
    def hide_typing(self):
        self.configure(state="normal")
        self.delete("typing.first", "typing.last")
        self.configure(state="disabled")