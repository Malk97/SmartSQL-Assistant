from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from transformers import AutoModel, AutoTokenizer
from langchain_groq import ChatGroq
from few_shots_prod import few_shots_prod  # Ensure this import works
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    """Initialize database connection with enhanced error handling"""
    try:
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None



def preprocess_few_shots_prod(few_shots_prod):
    """Convert `few_shots` data into string-compatible format for embedding."""
    processed_data = []
    for example in few_shots_prod:
        processed_example = {}
        for key, value in example.items():
            # Convert lists or dicts into strings
            if isinstance(value, list):
                if all(isinstance(item, dict) for item in value):  # Handle list of dictionaries
                    processed_value = "; ".join(
                        ", ".join(f"{k}: {v}" for k, v in item.items()) for item in value
                    )
                else:
                    processed_value = ", ".join(map(str, value))
            else:
                processed_value = str(value)  # Ensure all values are strings
            processed_example[key] = processed_value
        processed_data.append(processed_example)
    return processed_data


def get_embeddings_and_vectorstore(few_shots_prod):
    """
    Create embeddings and vectorstore using HuggingFace and FAISS.
    Preprocess the few_shots data to ensure compatibility.
    """
    try:
        processed_few_shots_prod = preprocess_few_shots_prod(few_shots_prod)

        # Combine all fields into a single text string for each example
        texts = [" ".join(example.values()) for example in processed_few_shots_prod]

        # Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name='distilbert-base-nli-mean-tokens')

        # Create FAISS vectorstore
        vectorstore = FAISS.from_texts(
            texts=texts, 
            embedding=embeddings, 
            metadatas=processed_few_shots_prod
        )

        return vectorstore
    except Exception as e:
        print(f"Embedding Creation Error: {e}")
        return None


def get_sql_chain(db, vectorstore):
    """Create SQL generation chain with improved prompt"""
    template = """
    You are a MySQL expert. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account and give my the query .
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Use API key from environment variable
    llm = ChatGroq(
        model="mixtral-8x7b-32768", 
        temperature=0, 
        api_key=os.getenv("GROQ_API_KEY")
    )

    def get_schema(_):
        try:
            return db.get_table_info()
        except Exception as e:
            st.error(f"Schema Retrieval Error: {e}")
            return ""

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabaseChain, chat_history: list, vectorstore):
    """Generate response with comprehensive error handling"""
    try:
        sql_chain = get_sql_chain(db, vectorstore)
        
        template = """
            You are a MySQL expert. You are interacting with a user who is asking you questions about the company's database.
            Based on the table schema below, question, sql query, and sql response, write a natural language response and put the output in table if that need the answer only.
            <SCHEMA>{schema}</SCHEMA>

            Conversation History: {chat_history}
            SQL Query: <SQL>{query}</SQL>
            User question: {question}
            SQL Response: {response}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGroq(
            model="mixtral-8x7b-32768", 
            temperature=0, 
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]) if vars.get("query") else "No query generated"
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
    except Exception as e:
        st.error(f"Response Generation Error: {e}")
        return f"An error occurred: {e}"

def main():
    st.set_page_config(page_title="Query Quokka", page_icon=":robot_face:")
    st.title("🤖 Query Quokka")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm a Query Quokka assistant. Connect to a database and start asking questions.")
        ]

    # Database connection sidebar
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        # Use environment variables as defaults if available
        host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"))
        port = st.text_input("Port", value=os.getenv("DB_PORT", "3306"))
        user = st.text_input("Username", value=os.getenv("DB_USER", "root"))
        password = st.text_input("Password", type="password")
        database = st.text_input("Database", value='atliq_tshirts')
        
        if st.button("Connect to Database"):
            with st.spinner("Connecting..."):
                db = init_database(user, password, host, port, database)
                if db:
                    vectorstore = get_embeddings_and_vectorstore(few_shots_prod)
                    st.session_state.db = db
                    st.session_state.vectorstore = vectorstore
                    st.success("Connected successfully!")
                else:
                    st.error("Connection failed. Check your details.")

    # Chat interface
    if "db" not in st.session_state:
        st.warning("Please connect to a database first.")
    else:
        # Render existing chat history
        for message in st.session_state.chat_history:
            with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
                st.markdown(message.content)

        # User input
        if user_query := st.chat_input("Ask a question about your database"):
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            with st.chat_message("Human"):
                st.markdown(user_query)
            
            with st.chat_message("AI"):
                response = get_response(
                    user_query, 
                    st.session_state.db, 
                    st.session_state.chat_history, 
                    st.session_state.vectorstore
                )
                st.markdown(response)
            
            st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()