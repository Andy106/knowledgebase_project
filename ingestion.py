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
    api_key="******",
    environment="gcp-starter",
)
  
directory = 'C:\\D Drive\\Software Engineering\\LLM\knowledgebase_project\\mediumblogs'
 
# iterate over files in that directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        f = os.path.join(root, filename)

        loader = TextLoader(f, encoding = 'UTF-8')
        document = loader.load()

        # print(document)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)
        print(len(texts))

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    ) 
