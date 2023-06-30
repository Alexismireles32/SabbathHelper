import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
headers = {
    "authorization": st.secrets["OPENAI_API_KEY"],
    "content-type": "application/json"
}
# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader('Manual_iglesia.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='Manual_iglesia')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="Manual_iglesia",
    description="A church manual in pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('🦜🔗 El Manual de Iglesia para Adventistas del Septimo Dia')
# Create a text input box for the user
prompt = st.text_input('Pregunta cualquier cosa al Manual de Iglesia y tendras una respuesta precisa y util! Te aconsejo que pidas el capitulo y la pagina de la infromacion en tu pregunta')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 
