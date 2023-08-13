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
)  # Tkinter Python Module for GUI


class GUI:
    client_socket = None
    last_received_message = None

    def __init__(self, master):
        self.root = master
        self.chat_transcript_area = None
        self.name_widget = "Driver"
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
        self.chat_transcript_area = Text(frame, width=70, height=10, font=("Serif", 12))
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
        self.model_.set("llama2")
        models = ["llama2", "gpt3.5-turbo"]
        self.model_dropdown = OptionMenu(
            frame2,
            self.model_,
            *models,
        )
        self.model_dropdown.pack(side="left", padx=10)

        frame2.pack(side="top", pady=10)
        frame.pack(side="top")

    def display_chat_entry_box(self):
        frame = Frame()
        Label(frame, text="Enter message:", font=("Serif", 12)).pack(
            side="top", anchor="w"
        )
        self.enter_text_widget = Text(frame, width=60, height=3, font=("Serif", 12))
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
        self.chat_transcript_area.insert("end", message.decode("utf-8") + "\n")
        self.chat_transcript_area.yview(END)
        self.enter_text_widget.delete(1.0, "end")
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
