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

## Example Data Cleaning Code

```python
import pandas as pd
from dateutil.parser import parse
from datetime import datetime

E1 = pd.read_csv("employee_data.csv")

E1['ExitDate'] = E1['ExitDate'].apply(lambda x: parse(x, dayfirst=True) if pd.notnull(x) else None)
current_date = datetime.now().strftime('%Y-%m-%d')
E1['ExitDate'].fillna(current_date, inplace=True)

E1['TerminationDescription'].fillna("Still Working", inplace=True)
E1.loc[E1['TerminationType'] == 'Unk', 'TerminationType'] = 'Still Working'

E1['StartDate'] = E1['StartDate'].apply(lambda x: parse(x, dayfirst=True) if pd.notnull(x) else None)

E1['Age'] = E1['DOB'].apply(lambda x: parse(x, dayfirst=True).year if pd.notnull(x) else None)
current_year = pd.Timestamp('today').year
E1['Age'] = current_year - E1['Age']
E1.drop('DOB', axis=1, inplace=True)

business_unit_mapping = {
    'BPC': 'Business Planning and Control',
    # ... other mappings
}
E1['BusinessUnit'] = E1['BusinessUnit'].map(business_unit_mapping).fillna(E1['BusinessUnit'])

state_mapping = {
    'AL': 'Alabama',
    # ... other mappings
}
E1['State'] = E1['State'].map(state_mapping).fillna(E1['State'])

payzone_mapping = {
    'Zone A': 'Entry-Level Salary',
    'Zone B': 'Mid-Level Salary',
    'Zone C': 'Senior-Level Salary'
}
E1['PayZone'] = E1['PayZone'].map(payzone_mapping).fillna(E1['PayZone'])

E1_part1 = E1[['EmpID', 'FirstName', 'LastName', 'StartDate', 'Title', 'Supervisor', 'ADEmail', 'BusinessUnit',
               'EmployeeStatus', 'EmployeeType', 'PayZone', 'EmployeeClassificationType', 'DepartmentType',
               'Division', 'State', 'JobFunctionDescription', 'Gender', 'LocationCode', 'Race_Desc',
               'Marital_Desc', 'Performance_Score', 'Current_Employee_Rating', 'Age']]

E1_part2 = E1[['EmpID', 'ExitDate', 'TerminationType', 'TerminationDescription']]
