import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone # This is the pinecone client

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

pinecone.init(
    api_key="*****",
    environment="gcp-starter",
)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

docsearch = Pinecone.from_existing_index(
        index_name="medium-blogs-embeddings-index", embedding=embeddings
    )

qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

query = "Which service can I use to observe inter service communication in an ECS cluster?"
result = qa({"query": query})
<<<<<<< HEAD
print(result)
#print(result['result'])
=======
print(result['result'])
>>>>>>> 1d98f25f5bca6a998f9dec2705137fe7edee9879
