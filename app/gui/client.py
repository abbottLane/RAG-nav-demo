from tkinter import (
    Tk,
    Frame,
    Scrollbar,
    Label,
    END,
    Entry,
    Text,
    VERTICAL,
    Button,
    messagebox,
    StringVar,
    OptionMenu,
)
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from conversation.openai import OpenAIConversation
from nlp.classifiers import zero_shot_classification, is_recommendation_request

CONVERSATION_AGENT = OpenAIConversation()


class GUI:
    client_socket = None
    last_received_message = None

    def __init__(self, master):
        self.root = master
        self.chat_transcript_area = None
        self.name_widget = "HUMAN"
        self.enter_text_widget = None
        self.join_button = None
        self.location_ = None
        self.initialize_gui()

    def initialize_gui(self):  # GUI initializer
        self.root.title("RAG Chat")
        # Make it so the window can be resized
        self.root.resizable(width=False, height=False)

        self.display_chat_box()  # display chat box
        self.display_chat_entry_box()  # display chat entry box

    def display_chat_box(self):
        frame = Frame()
        Label(frame, text="Transcript:", font="Serif 12 bold", pady=10).pack(
            side="top", anchor="w"
        )
        self.chat_transcript_area = Text(
            frame,
            width=70,
            height=40,
            font=("Serif", 12),
            bg="black",
            fg="white",
            wrap="word",
        )
        scrollbar = Scrollbar(
            frame, command=self.chat_transcript_area.yview, orient=VERTICAL
        )
        self.chat_transcript_area.config(yscrollcommand=scrollbar.set)
        self.chat_transcript_area.bind("<KeyPress>", lambda e: "break")
        self.chat_transcript_area.pack(side="left", padx=20)
        scrollbar.pack(side="left", fill="y")

        frame2 = Frame()
        Label(frame2, text="Location:", padx=10, font=("Serif", 12)).pack(
            side="left", anchor="w"
        )
        self.location_ = StringVar(frame2)
        self.location_.set("Duvall")
        locations = ["Duvall, WA", "Georgetown, DC"]
        self.location_dropdown = OptionMenu(
            frame2,
            self.location_,
            *locations,
        )
        self.location_dropdown.pack(side="left", padx=10)

        Label(frame2, text="Model:", padx=10, font=("Serif", 12)).pack(
            side="left", anchor="w"
        )
        self.model_ = StringVar(frame2)
        self.model_.set("gpt-3.5-turbo")
        models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]
        self.model_dropdown = OptionMenu(
            frame2,
            self.model_,
            *models,
        )
        self.model_dropdown.pack(side="left", padx=10)

        Label(frame2, text="API Key: ", padx=10, font=("Serif", 12)).pack(
            side="left", anchor="w"
        )
        # text input box for api key
        self.api_key = Entry(frame2, width=20, show="x", font=("Serif", 12))
        self.api_key.pack(side="left", padx=10)

        frame2.pack(side="top", pady=10)
        frame.pack(side="top")

    def display_chat_entry_box(self):
        frame = Frame()
        Label(frame, text="Enter message:", font=("Serif", 12)).pack(
            side="top", anchor="w"
        )
        self.enter_text_widget = Text(
            frame,
            width=60,
            height=3,
            font=("Serif", 12),
            wrap="word",
            bg="black",
            fg="white",
        )
        self.enter_text_widget.pack(side="left", pady=15)
        self.enter_text_widget.bind("<Return>", self.on_enter_key_pressed)
        frame.pack(side="top")

    def on_enter_key_pressed(self, event):
        self.send_chat()
        self.clear_text()

    def clear_text(self):
        self.enter_text_widget.delete(1.0, "end")

    def send_chat(self):
        senders_name = self.name_widget
        data = self.enter_text_widget.get(1.0, "end").strip()
        message = (senders_name + ": " + data).encode("utf-8")
        self.chat_transcript_area.insert("end", message.decode("utf-8") + "\n\n")
        self.chat_transcript_area.yview(END)
        self.enter_text_widget.delete(1.0, "end")
        #  flush the gui
        self.root.update()

        # detect whether the message is a request for local recommendations
        if is_recommendation_request(message.decode("utf-8")):
            # do vectordb search
            location = self.location_.get()

        # send message and context to openai
        CONVERSATION_AGENT.set_api_key(self.api_key.get())
        response = asyncio.run(
            CONVERSATION_AGENT.chatgpt_response(
                message.decode("utf-8").lstrip(self.name_widget + ": "),
                self.model_.get(),
            )
        )

        # write response in green text
        self.chat_transcript_area.insert("end", "AI: " + response + "\n\n")
        self.chat_transcript_area.yview(END)

        return "break"

    def on_close_window(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            self.client_socket.close()
            exit(0)


# the mail function
if __name__ == "__main__":
    root = Tk()
    gui = GUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_close_window)
    root.mainloop()
