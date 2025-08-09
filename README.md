# Query Quokka - AI-Powered SQL Query Assistant

Query Quokka is an interactive Streamlit application that allows users to connect to a MySQL database and ask natural language questions about their data. The application translates user queries into optimized SQL statements, executes them, and returns clear, human-readable answers.

---

## Features

- **Database Connection:** Easily connect to any MySQL database using credentials entered in the sidebar.
- **Natural Language to SQL:** Converts user questions into SQL queries leveraging an advanced language model (`ChatGroq`).
- **Few-Shot Learning:** Incorporates example questions and answers to guide query generation and improve accuracy.
- **Embeddings and Vectorstore:** Uses HuggingFace embeddings and FAISS vectorstore for semantic similarity on few-shot data.
- **Conversational UI:** Maintains chat history for contextual, multi-turn conversations.
- **Robust Error Handling:** Gracefully handles errors in database connection, schema retrieval, embedding generation, and response formation.
- **Natural Language Responses:** Returns SQL results as natural language, including formatted tables when applicable.

---

## Code Overview

### Database Initialization (`init_database`)

Establishes a connection to a MySQL database using provided credentials with error handling to notify users on failure.

### Data Preprocessing (`preprocess_few_shots_prod`)

Processes few-shot example data by converting lists and dictionaries into plain string format suitable for embedding creation.

### Embeddings & Vectorstore (`get_embeddings_and_vectorstore`)

Creates semantic embeddings from few-shot examples using HuggingFace’s `distilbert-base-nli-mean-tokens` model, then builds a FAISS vectorstore for efficient similarity search.

### SQL Chain Creation (`get_sql_chain`)

Generates a prompt that, combined with the current database schema and conversation history, guides the AI to produce accurate SQL queries without extra formatting.

### Response Generation (`get_response`)

Uses the generated SQL query to fetch data from the database and prompts the AI to generate a natural language answer based on the query results and conversation context.

### Main Application (`main`)

Runs the Streamlit app with a sidebar for database connection and a chat interface for users to ask questions and receive AI-generated responses.

---

## Data Cleaning Process

The employee dataset (`employee_data.csv`) was cleaned using the following steps:

- **Handling Missing Values:** Filled missing `ExitDate` with the current date for active employees and replaced missing `TerminationDescription` with `"Still Working"`. Changed `"Unk"` in `TerminationType` to `"Still Working"`.
- **Date Conversion:** Parsed date strings into `datetime` objects to enable date operations.
- **Age Calculation:** Calculated age from the `DOB` field by subtracting birth year from the current year, then dropped the original `DOB` column.
- **Categorical Mappings:** Expanded abbreviations in categorical columns (`BusinessUnit`, `State`, `PayZone`) using predefined dictionaries for clarity.
- **Data Splitting:** Divided the cleaned dataset into general employee info and termination-related info for easier analysis.
- **Validation:** Confirmed no remaining missing values and validated data types.

---

## Dependencies
- Python 3.10
- Streamlit
- LangChain
- HuggingFace Transformers
- FAISS
- mysql-connector-python
- Groq API client






