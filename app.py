import os
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo

os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI(temperature=0.5)
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('Manual_iglesia.pdf')

pages = loader.load_and_split()
page_content = [page[1] for page in pages]  # Extraer el contenido de las páginas
data = pd.DataFrame({'page_content': page_content})

store = Chroma.from_documents(data['page_content'], embeddings, collection_name='Manual_iglesia')

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
prompt = st.text_input('¿Qué pregunta tienes? Habla directamente con el manual y obtendrás respuestas precisas y útiles. Consejo: Especifica el capítulo y la página donde se encuentra la información.')

if prompt:
    response = agent_executor.run(f"{prompt}?")
    st.write(response)
