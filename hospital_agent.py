from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

from langchain.tools import Tool
from langchain.agents import initialize_agent

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings()

with open("data/medical_docs.txt", "r") as f:
    medical_text = f.readlines()
    
medical_db = FAISS.from_texts(medical_text, embeddings)
medical_retriever = medical_db.as_retriever()

with open("data/policy_docs.txt", "r") as f:
    policy_text = f.readlines()
    
policy_db = FAISS.from_texts(policy_text, embeddings)
policy_retriever = policy_db.as_retriever()


query = "What are dengue symptoms?"

docs = medical_retriever.get_relevant_documents(query)

print("Retrieved Medical Doc:")
print(docs[0].page_content)




def medical_rag(query):

    docs = medical_retriever.get_relevant_documents(query)

    context = docs[0].page_content

    return f"Medical Information: {context}"


def policy_rag(query):

    docs = policy_retriever.get_relevant_documents(query)

    context = docs[0].page_content

    return f"Hospital Policy: {context}"


tools = [

    Tool(
        name="Medical_RAG",
        func=medical_rag,
        description="Use this tool for medical questions about diseases, symptoms, or treatments."
    ),

    Tool(
        name="Policy_RAG",
        func=policy_rag,
        description="Use this tool for hospital policies like visiting hours, insurance, or appointments."
    )

]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)


while True:

    query = input("\nAsk a question (type 'exit' to stop): ")

    if query.lower() == "exit":
        break

    response = agent.run(query)

    print("\nFinal Answer:")
    print(response)
    

