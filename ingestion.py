import os
import pandas as pd

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
    api_key="5276d92e-17bc-46d5-865d-86977a64a8de",
    environment="gcp-starter",
)

# Create dictionary of source URLs
df = pd.read_csv("C:\\D Drive\\Software Engineering\\LLM\\knowledgebase_project\\source.csv")
dict1 = dict(df.values)

# Reading text files from the directory
directory = 'C:\\D Drive\\Software Engineering\\LLM\knowledgebase_project\\mediumblogs'


for root, dirs, files in os.walk(directory):
    for filename in files:
        f = os.path.join(root, filename)

        loader = TextLoader(f, encoding = 'UTF-8')
        document = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)

        for doc in texts:
            new_url = dict1[filename]
            doc.metadata.update({"source": new_url})

        print(f"Going to add {len(texts)} documents to Pinecone")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    ) 
