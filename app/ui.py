import gradio
import app.rag as rag
import app.inference as inference


system_prompt = "You're a helpful AI assistant that will help in providing explanation of user inquiry about a certain open source project. \
You will be given a couple of relevant parts of the source code, but you're free to use your knowledge as well. If there's a certain classess \
/ file / method you need to take a look before providing the answer\
 you can ask the user by saying:  To give you a more accurate answer, it would help very much to provide the content of SomeClass / SomeFile. You should response in Markdown"

user_visible_chat_history: list[dict[str, str]] = []
ai_visible_chat_history: list[dict[str, str]] = [{"role":"system", "content":system_prompt}]
last_user_message = ""

def chat():
    global user_visible_chat_history, ai_visible_chat_history, last_user_message
    response = inference.inquiry_ai(last_user_message, ai_visible_chat_history)
    similar_items = inference.get_current_similar_items()
    extra_context = ""
    user_visible_chat_history.append({"role":"assistant", "content":response})

    for item in similar_items:
        extra_context += item + "\n=================================================="

    return user_visible_chat_history, extra_context

def do_entry(message):
    global user_visible_chat_history, last_user_message
    user_visible_chat_history.append({"role":"user", "content":message})
    last_user_message = message
    return "", user_visible_chat_history


def do_embedding(source: str, db_path: str):
    rag.create_rag_database(source, db_path)
    inference.initialize(db_path)
    return "Embedding completed"

def load_embedding(db_path: str):
    print("Loading embedding from " + db_path)
    inference.initialize(db_path)
    return "Embedding loaded"


def launch_ui():
    with gradio.Blocks() as ui:
        with gradio.Row():
            source  = gradio.Textbox(label="The Parent Directory of Your Source Code")
            db_path = gradio.Textbox(label="The Path to Store Embedding")
        with gradio.Row():
            do_embedding_btn = gradio.Button("Run Embedding Process")
        with gradio.Row():
            load_db_path = gradio.Textbox(label="Load Existing Embedding")
        with gradio.Row():
            load_embedding_btn = gradio.Button("Load Embedding")
        with gradio.Row():
            embedding_log = gradio.Textbox(label="Status (embedding process could take long time, see log for more details!)")
        with gradio.Row():
            chat_bot       = gradio.Chatbot(type="messages", height=500)
            extra_context  = gradio.TextArea(label='Behind the scene context provided to the bot:', lines=22)
        with gradio.Row():
            entry = gradio.Textbox(label="Inquiry the AI Assistant About the Project:")
        with gradio.Row():
            clear = gradio.Button("Clear")

         

        entry.submit(do_entry, inputs=[entry], outputs=[entry, chat_bot]).then(
            chat, inputs=None, outputs=[chat_bot, extra_context]
        )
        do_embedding_btn.click(fn=do_embedding, inputs=[source, db_path], outputs=[embedding_log], queue=False)
        load_embedding_btn.click(fn=load_embedding, inputs=[load_db_path], outputs=[embedding_log], queue=False)

    ui.launch()