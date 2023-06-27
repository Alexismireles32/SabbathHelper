import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo

os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI(temperature=0.5)  # Adjust temperature if needed
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('Manual_iglesia.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='Manual_iglesia')

vectorstore_info = VectorStoreInfo(
    name="Manual_iglesia",
    description="El Manual de la iglesia",
    vectorstore=store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit
)
st.title('🦜🔗 El manual de la Iglesia Adventista del Septimo Dia')
prompt = st.text_input('¿Que pregunta tienes? Habla directamente con el manual y obtendras respuestas precisas y utiles! Consejo: Especifica que quieres saber el Capitulo y la pagina donde se encuentra la infromacion')

if prompt:
    response = agent_executor.run(f"{prompt}?")
    st.write(response)
