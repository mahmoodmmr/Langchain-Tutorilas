from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "sk**"
llm_name = "gpt-3.5-turbo"

app = Flask(__name__)


class ChatWithYourDataBot:
    def __init__(self):
        self.loaded_file = (
            "G:\\Desktop\\Simple Sample\\AiHealth\\Job 4\\nutrirtion_nemone_Diet.pdf"
        )
        self.qa = self.load_db(self.loaded_file, "stuff", 4)

    def load_db(self, file, chain_type, k):
        loader = PyPDFLoader(file)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa

    def _convert_document_to_dict(self, document):
        # This is a generic method; adjust based on your Document class structure
        document_dict = {}

        # Assuming Document has a 'metadata' attribute
        document_dict["metadata"] = document.metadata

        # Assuming Document has a 'get_content()' method
        if hasattr(document, "get_content"):
            document_dict["content"] = document.get_content()
        # If not, you might need to adjust this based on the actual structure

        # Add other attributes or methods as needed

        return document_dict

    def ask_question(self, query):
        result = self.qa({"question": query, "chat_history": []})
        # Convert Document objects to JSON-serializable format
        result["source_documents"] = [
            self._convert_document_to_dict(doc) for doc in result["source_documents"]
        ]
        return result


chat_bot = ChatWithYourDataBot()


@app.route("/ask", methods=["POST"])
def ask_question_api():
    data = request.get_json()
    query = data.get("question")
    response = chat_bot.ask_question(query)
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
