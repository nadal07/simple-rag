from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate


set_debug(True)
set_verbose(True)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_mode: str = "deepseek-r1:14b"):
        self.model = ChatOllama(model=llm_mode)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "you are a helpful assistant that can answer questions related to investment about the CSV document that was uploaded by the user.",

                ),
                (
                    "human",
                    "Here is the document pieces: {context}\n Question: {question}",
                )
            ]
        )
        self.vector_store = None
        self.retriever = None
        self.chain = None

    
    def ingest(self, pdf_file_path: str):
        if pdf_file_path.endswith(".csv"):
            docs = CSVLoader(file_path=pdf_file_path).load()
        if pdf_file_path.endswith("pdf"):
            docs = PyPDFLoader(file_path=pdf_file_path).load()
        else :
            docs = CSVLoader(file_path=pdf_file_path).load()

        print(docs)
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )
        # self.vector_store = Chroma.from_documents(
        #     documents=chunks,
        #     embedding=SentenceTransformerEmbeddings(model_name="all-minilm:l6-v2"),
        #     persist_directory="chroma_db",
        # )

    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
            )
        # if not self.vector_store:
        #     self.vector_store = Chroma(
        #         persist_directory="chroma_db", embedding=SentenceTransformerEmbeddings(model_name="all-minilm:l6-v2")
        #     )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 20, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

