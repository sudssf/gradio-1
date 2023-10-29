import gradio as gr
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings
import chromadb

from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        #chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    #".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
chunk_size = 500
chunk_overlap = 50
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def process_documents(file_path: str) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {file_path}")
    documents = load_single_document(file_path)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    print(f"Checks if vectorstore exists at {persist_directory}")
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    sqllite_files = glob.glob(os.path.join(persist_directory, '*.sqlite3'))
    if len(sqllite_files) > 1:
        return True
    sqllite_files = glob.glob(os.path.join(persist_directory, '*'))
    if len(sqllite_files) > 1:
        return True
    return False

def process_upload_file(file_path: str) :
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectorstore = Chroma("langchain_store", embeddings)
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        document_collection = chroma_client.get_or_create_collection(name=persist_directory)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                     #client_settings=CHROMA_SETTINGS
                     )
        
        #collection = db.get()
        texts = process_documents(file_path=file_path)
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
        # print(texts)
        #document_collection.add(texts)
    else:
        # Create and store locally vectorstore
        print(f"Creating new vectorstore at {persist_directory}")
        texts = process_documents(file_path=file_path)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, 
                                   persist_directory=persist_directory, 
                                   #client_settings=CHROMA_SETTINGS
                                   )
        # #db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        # chroma_client = chromadb.PersistentClient(path='db')
        # document_collection = chroma_client.get_or_create_collection(name=persist_directory)
        # #document_collection.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
        # document_collection.add(
        #     embeddings=embeddings,
        #     metadatas=texts,
        # )
        #print(texts)
        #document_collection.add(texts)
    db.persist()
    #db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

def upload_file(file):
    try:
        process_upload_file(file.name)
    except Exception as e:
        print(e)
    return file.name


workspaces = ["workspace1", "workspace2", "workspace3"]
llm_chain = None
llm = None
def selected(name):
    global PERSIST_DIRECTORY
    global persist_directory
    global llm_chain
    persist_directory = name
    PERSIST_DIRECTORY = name
    CHROMA_SETTINGS = Settings(
        #chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
    )
    llm_chain, llm = init_chain()
    return f"Workspace: {name}!"

def add_workspace(workspace_name: str):
    global workspaces
    if workspace_name and workspace_name not in workspaces:
        workspaces.append(workspace_name)
        return f"{workspace_name} added!"
    else:
        return f"Current workspaces are: {workspaces}"
def select_workspace(workspace_name: str):
    return f"You selected: {workspace_name}"

def initialize_model_and_tokenizer(model_name="bigscience/bloom-1b7"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def init_chain(): #model, tokenizer):
    # class CustomLLM(LLM):

    #     def _call(self, prompt, stop=None, run_manager=None) -> str:
    #         inputs = tokenizer(prompt, return_tensors="pt")
    #         result = model.generate(input_ids=inputs.input_ids, max_new_tokens=20)
    #         result = tokenizer.decode(result[0])
    #         return result

    #     @property
    #     def _llm_type(self) -> str:
    #         return "custom"

    #llm = CustomLLM()
    callbacks = [] #if args.mute_stream else [StreamingStdOutCallbackHandler()]
    model = os.environ.get("MODEL", "mistral")
    llm = Ollama(model=model, callbacks=callbacks)
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
                #, client_settings=CHROMA_SETTINGS
                )
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    print(f"creating QA chain using {persist_directory}")
    llm_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
    # template = """Question: {question}
    # Answer: Let's think step by step."""
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm
def user(user_message, history):
    return "", history + [[user_message, None]]
def doc_to_str(document):
    return "\n> " + document.metadata["source"] + ":" + (document.page_content)
def bot(history):
    if llm_chain is None:
        return "select workspace!"
    
    question = history[-1][0]
    print("Question: ", history[-1][0])
    #bot_message = llm_chain.run(question=history[-1][0])
    result = llm_chain({"query": question})
    bot_message = result["result"]
    if result["source_documents"]:
        bot_message += "\n references: " + "\n".join(map( lambda x: doc_to_str(x), result["source_documents"]))
    print("Response: ", bot_message)
    history[-1][1] = ""
    history[-1][1] += bot_message
    return history

with gr.Blocks() as demo:
   with gr.Row() as main:
        with gr.Column(scale=0.2, variant='panel', min_width=200):
            with gr.Row():
                workspace_select = gr.Dropdown(choices=workspaces, label="Select workspace")
                #output = gr.Textbox(label="Output Box")
                select_btn = gr.Button("Select")
            with gr.Row():
                file_upload = gr.File(label="Uplaod file")
                upload_button = gr.UploadButton("Click to Upload a File")
            
            with gr.Row():
                addw = gr.Textbox(label="Add workspace")
                greet_btn = gr.Button("Add")
                

        with gr.Column(scale=2, min_width=800):
            # img1 = gr.Image("images/cheetah.jpg")
            output = gr.Label(label="Output Box")
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Message")
            clear = gr.Button("Clear")
            

            
            btn = gr.Button("Go")
        select_btn.click(fn=selected, inputs=workspace_select, outputs=output)
        workspace_select.change(fn=selected, inputs=workspace_select, outputs=output)
        greet_btn.click(fn=add_workspace, inputs=addw, outputs=output)
        upload_button.upload(upload_file, upload_button, output)

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
        
demo.launch()