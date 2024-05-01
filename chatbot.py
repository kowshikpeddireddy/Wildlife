import tkinter as tk
from tkinter import ttk
import openai

# Set your OpenAI API key here
api_key = "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"

# Function to start the chatbot
def chatbot():
    # Create a list to store all the messages for context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def send_message(event=None):
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)
        messages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            api_key=api_key
        )

        chat_message = response['choices'][0]['message']['content']
        conversation_text.insert(tk.END, f"User: {user_input}\nBot: {chat_message}\n")
        messages.append({"role": "assistant", "content": chat_message})

        if user_input.lower() == "quit":
            user_entry.config(state=tk.DISABLED)

    conversation_text = tk.Text(root, wrap=tk.WORD, bg="#F0F0F0", bd=0, font=("Arial", 12), padx=10, pady=10)
    conversation_text.pack(expand=True, fill="both")

    user_entry = tk.Entry(root, width=50, bd=0, font=("Arial", 12), bg="#FFFFFF")
    user_entry.pack(pady=10)
    user_entry.bind("<Return>", send_message)

    send_button = tk.Button(root, text="Send", command=send_message, bg="#4CAF50", fg="#FFFFFF", width=30, bd=0, font=("Arial", 12, "bold"))
    send_button.pack(pady=10)


root = tk.Tk()
root.title("Chatbot and Questions Generator")
root.geometry("800x600")
root.configure(bg="#FFFFFF")

# Create a frame for the buttons
button_frame = tk.Frame(root, bg="#FFFFFF")

# Function to create styled buttons
def create_styled_button(frame, text, command, width, height):
    button = tk.Button(frame, text=text, command=command, width=width, height=height, bg="#4CAF50", fg="#FFFFFF", bd=0, font=("Arial", 12, "bold"))
    return button

# Create styled buttons for Chatbot and Generate Questions
chatbot_button = create_styled_button(button_frame, "Open Chatbot", chatbot, width=30, height=3)

# Pack the buttons in the frame
chatbot_button.pack(pady=10)

# Place the button_frame at the center of the window
button_frame.place(relx=0.5, rely=0.5, anchor="center")

root.mainloop()
