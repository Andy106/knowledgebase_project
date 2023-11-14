import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain # Use ConversationalRetrievalChain instead of RetrievalQA chain when we need to use memory
import pinecone # This is the pinecone client

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

pinecone_api_key = os.getenv('PINECONE_API_KEY')

pinecone.init(
    api_key=pinecone_api_key,
    environment="gcp-starter",
)

def run_llm(query, chat_history):

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    docsearch = Pinecone.from_existing_index(
        index_name="medium-blogs-embeddings-index", embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm( # Note from_llm function instead of from_chain_type. This is how the ConversationalRetrievalChain package has been implemented
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    # query = "Which service can I use to observe inter service communication in an ECS cluster?"
    result = qa({"question": query, "chat_history": chat_history})

    return(result)
