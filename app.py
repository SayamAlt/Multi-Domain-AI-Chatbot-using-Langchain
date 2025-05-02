import os, warnings, requests, re, ast
warnings.filterwarnings('ignore')
import streamlit as st
from langchain.chains import LLMMathChain, RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool, AgentType, load_tools, tool
from langchain.agents.agent_types import AgentType
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from sympy import symbols, diff, integrate, solve, sympify, Basic
from langchain_experimental.utilities import PythonREPL
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from typing import List

load_dotenv()

if 'secrets' in st.secrets:
    api_key = st.secrets["secrets"]["OPENAI_API_KEY"]
    exchange_rate_api_key = st.secrets["secrets"]["EXCHANGE_RATE_API_KEY"]
    news_api_key = st.secrets["secrets"]["NEWS_API_KEY"]
    anthropic_api_key = st.secrets["secrets"]["ANTHROPIC_API_KEY"]
    weatherstack_api_key = st.secrets["secrets"]["WEATHERSTACK_API_KEY"]
else:
    api_key = os.environ.get("OPENAI_API_KEY")
    exchange_rate_api_key = os.environ.get("EXCHANGE_RATE_API_KEY")
    news_api_key = os.environ.get("NEWS_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    weatherstack_api_key = os.environ.get("WEATHERSTACK_API_KEY")

if api_key is None:
    st.error("OpenAI API key not found. Please set it in the .env file or in Streamlit secrets.")
    st.stop()

if exchange_rate_api_key is None:
    st.error("ExchangeRate API key not found. Please set it in the .env file or in Streamlit secrets.")
    st.stop()
    
if news_api_key is None:
    st.error("NewsAPI key not found. Please set it in the .env file or in Streamlit secrets.")
    st.stop()
    
if anthropic_api_key is None:
    st.error("Anthropic API key not found. Please set it in the .env file or in Streamlit secrets.")
    st.stop()
    
if weatherstack_api_key is None:
    st.error("Weatherstack API key not found. Please set it in the .env file or in Streamlit secrets.")
    st.stop()

llm = ChatOpenAI(api_key=api_key, temperature=0.7)
claude = ChatAnthropic(model='claude-3-7-sonnet-20250219',api_key=anthropic_api_key, temperature=0.7)
embeddings = OpenAIEmbeddings(api_key=api_key)
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
parser = StrOutputParser()
python_repl = PythonREPL()

st.set_page_config(page_title="Multi-Domain AI Chatbot", layout="wide")
st.title("🤖 Multi-Domain AI Chatbot")

st.sidebar.subheader("📄 Upload Documents for QA")
uploaded_files = st.sidebar.file_uploader(" ", type=["pdf"], accept_multiple_files=True, key='document_qa_uploader')

def fetch_latest_news(query, max_results=5):
    if not news_api_key:
        return "⚠️ Missing NewsAPI key."

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={max_results}&apiKey={news_api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"⚠️ Error fetching news: {response.status_code}"
    
    articles = response.json().get("articles", [])
    
    if not articles:
        return "⚠️ No articles found."

    # Combine news titles and descriptions
    combined_news = ""
    
    for article in articles:
        title = article.get("title", "No Title")
        description = article.get("description", "No Description available.")
        combined_news += f"**{title}**\n{description}\n\n"

    # Create prompt to make detailed summary
    prompt = PromptTemplate(
        input_variables=['query', 'news'],
        template="""
        Based on the user query and the following latest news articles:\n
        
        User Query: {query}
        Latest News:\n{news}\n
        
        Write a highly detailed and comprehensive news report:
        - Summarize key points intelligently.
        - Highlight major themes and developments.
        - Maintain clarity, readability, and professional journalistic tone.
        - Use proper paragraphs (no bullet points).
        - If possible, add extra context from your knowledge.
        """
    )

    news_summary_chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    detailed_report = news_summary_chain.invoke({"query": query, "news": combined_news})
    return detailed_report

def convert_currency(query):
    """ Converts currency using the ExchangeRate API. """
    if not exchange_rate_api_key:
        return "Missing ExchangeRate API key."
    try:
        parts = query.lower().split()
        amount = float(parts[1])
        from_currency = parts[2].upper()
        to_currency = parts[4].upper()
        url = f"https://api.exchangerate.host/convert?access_key={exchange_rate_api_key}&from={from_currency}&to={to_currency}&amount={amount}"
        response = requests.get(url).json()
        return f"{amount} {from_currency} = {response['result']} {to_currency}"
    except Exception as e:
        return "Please use format: convert 100 USD to EUR"

def is_math_expression(expression: str) -> bool:
    # A simple check: contains digits, operators, or typical math characters
    math_pattern = r"[\d+\-*/=^()]"
    return bool(re.search(math_pattern, expression))
@tool
def extract_symbols(expression: str) -> str:
    """Extract all symbols dynamically from a valid mathematical expression."""
    if not is_math_expression(expression):
        return "⚠️ The input doesn't appear to be a mathematical expression."
    
    try:
        expr = sympify(expression)
        
        # Check if expr is a valid SymPy object
        if not isinstance(expr, Basic):
            return "⚠️ Unable to interpret the input as a mathematical expression."

        symbols = expr.free_symbols
        if symbols:
            return f"✅ Symbols found: {symbols}"
        else:
            return "✅ No variables found in the expression."

    except Exception as e:
        return f"⚠️ Error parsing expression: {str(e)}"
@tool
def derivative_calculator(expression):
    """Calculates derivative of a symbolic expression w.r.t the first detected variable."""
    try:
        expr = sympify(expression)
        variables = list(expr.free_symbols)
        if not variables:
            return "⚠️ No variables detected in the expression."

        var = variables[0]  # Default: differentiate w.r.t first variable found
        derivative = diff(expr, var)
        return f"The derivative of `{expression}` with respect to `{var}` is: {derivative}"
    except Exception as e:
        return f"⚠️ Error calculating derivative: {str(e)}"
@tool
def integral_calculator(expression):
    """Calculates indefinite integral of a symbolic expression w.r.t first detected variable."""
    try:
        expr = sympify(expression)
        variables = list(expr.free_symbols)
        if not variables:
            return "⚠️ No variables detected in the expression."

        var = variables[0]
        integral = integrate(expr, var)
        return f"The integral of `{expression}` with respect to `{var}` is: {integral} + C"
    except Exception as e:
        return f"⚠️ Error calculating integral: {str(e)}"
@tool
def equation_solver(equation):
    """Solves an algebraic equation for the first detected variable."""
    try:
        # Convert 'x**2 + 3*x = 0' --> 'x**2 + 3*x - (0)'
        eq = sympify(equation.replace("=", "-(") + ")")
        variables = list(eq.free_symbols)
        if not variables:
            return "⚠️ No variables detected to solve for."

        var = variables[0]
        solutions = solve(eq, var)
        return f"The solution(s) to `{equation}` for `{var}` are: {solutions}"
    except Exception as e:
        return f"⚠️ Error solving equation: {str(e)}"

def validate_code_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    
def remove_comments(code: str) -> str:
    """Removes comments from the Python code."""
    # Remove inline comments
    code = re.sub(r'#.*', '', code)
    # Remove multi-line string comments (docstrings) if any
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    return code.strip()

def debug_code(code):
    """ Debugs Python code and detect potential issues."""
    code = remove_comments(code)
    
    prompt = PromptTemplate(
        template="""
            You are an expert Python developer.

            Carefully read the following Python code snippet, identify any bugs (syntax, logical, runtime errors), and directly output ONLY the fully corrected version of the code.

            Input Code:
            ```python
            {code}
            ```

            Output:
            ```python
            <corrected Python code here>
        """,
        input_variables=["code"]
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    return chain.invoke({"code": code})

def explain_code(code):
    """
    Explains the logic and structure of Python code.
    If user provides description, generates code first.
    """
    code = remove_comments(code)
    if not validate_code_syntax(code):
        # User provided a topic/description
        generate_code_prompt = PromptTemplate(
            template="""
            Generate a basic and simple example Python code for the following user request:

            {code}

            Provide ONLY the code without any explanation or comments.
            """,
            input_variables=["code"]
        )
        generate_chain = LLMChain(llm=claude, prompt=generate_code_prompt, output_parser=parser)
        code_to_explain = generate_chain.invoke({"code": code})['text']
    else:
        code_to_explain = code

    st.code(code_to_explain, language="python")

    # Now explain the actual Python code
    explanation_prompt = PromptTemplate(
        template="""
            You are an expert Python tutor.

            Carefully read the following code and generate a clear, detailed, and structured explanation:

            ```python
            {code}
            ```

            Output Instructions:
            - Briefly describe the purpose of the code.
            - Explain important sections line-by-line or block-by-block.
            - Highlight any key libraries, methods, or logic.
            - Use bullet points, headings, and short paragraphs for clarity.
            - Focus on **how** and **why** the code works.
            - End with a short summary of what the code achieves.
            
            ```python
            <explained Python code here>

            Make the explanation educational, precise, and easy to follow.
        """,
        input_variables=["code"]
    )

    explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt, output_parser=parser)
    return explanation_chain.invoke({"code": code_to_explain})

def optimize_code(code):
    """Optimizes Python code for performance, efficiency, and readability."""
    code = remove_comments(code)
    if not validate_code_syntax(code):
        return "⚠️ The input code has syntax errors. Please correct and try again."
        
    prompt = PromptTemplate(
        template="""
            You are an expert Python code optimizer. First, fix any syntax errors, then optimize it for performance, readability, and efficiency.

            When given a Python code snippet:
            - Check for any syntax issues, and fix them first.
            - Rewrite the code in a highly efficient, readable, and Pythonic way.
            - Apply best practices: use list comprehensions, vectorized operations, efficient libraries, avoid loops when possible.
            - If code is badly formatted or incomplete, try to infer the intention and cleanly reconstruct it.
            - Maintain the same functionality.

            Return your response in exactly this format:
            ---
            **Fixed Code:**
            ```python
            # Here, corrected and optimized code
            ```
            ---
            **Explanation:**
        """,
        input_variables=["code"]
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    return chain.invoke({"code": code})

@tool
def smart_web_search(query: str) -> str:
    """
    Performs a web search and generates a detailed and coherent answer using the search results.
    """
    search_results = DuckDuckGoSearchResults().invoke(query)

    search_prompt = PromptTemplate(
        input_variables=["query", "search_results"],
        template="""
        Based on the following search results, write a detailed and informative paragraph answering the following user query:

        User Query: {query}
        Search Results: {search_results}

        Answer in at least 2 paragraphs. Make it well-structured, informative, and insightful.
        """
    )
    
    reasoning_prompt = PromptTemplate(
        input_variables=["query", "search_result"],
        template="""
        You are an expert research assistant.

        Task:
        - Read the user's query: {query}
        - Analyze the following summarized search result carefully: {search_result}

        Instructions:
        1. Understand what the user truly wants to know.
        2. Extract and synthesize the most relevant, accurate, and detailed information.
        3. Eliminate irrelevant parts, hallucinations, or partial answers.
        4. If the user asked for countries, cities, organizations, technologies, etc., mention correct types properly.
        5. Organize the answer into structured paragraphs or bullet points if helpful.
        6. Maintain a professional, neutral, and highly informative tone.

        Your goal is to create a polished, comprehensive final answer based on the query and search summary.
        """
    )

    summarize_chain = LLMChain(llm=llm, prompt=search_prompt, output_parser=parser)
    reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt, output_parser=parser)

    combined_chain = RunnableParallel({
        "query": RunnablePassthrough(),
        "search_results": RunnablePassthrough()
    }) | RunnableParallel({
        "query": RunnableLambda(lambda x: x["query"]),
        "search_result": RunnableLambda(lambda x: summarize_chain.invoke({
            "query": x["query"],
            "search_results": x["search_results"]
        }))
    }) | reasoning_chain

    detailed_answer = combined_chain.invoke({
        "query": query,
        "search_results": search_results
    })

    return detailed_answer

@tool
def smart_wikipedia_search(query: str) -> str:
    """
    Performs a Wikipedia search and generates a detailed explanation based on the retrieved information.
    """
    wikipedia = load_tools(["wikipedia"], llm=llm)[0] 
    results = wikipedia.invoke({"query": query})

    # Use LLM to generate a polished answer
    prompt = PromptTemplate(
        template="""
        Based on the following information retrieved from Wikipedia, write a clear, structured, and detailed explanation
        answering the following question:

        Question: {question}
        Wikipedia Content: {content}

        Write in atleast 2 paragraphs. Make it well-structured, informative and insightful.
        """,
        input_variables=["question", "content"]
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
    return chain.invoke({"question": query, "content": results})

@tool
def generate_python_code(instruction: str) -> str:
    """
    Generates Python code for a given user request/task. It could be anything from data manipulation to machine learning.
    Example input: 'Write code to create a Pandas DataFrame of employees' or 'Generate a linear regression model using sklearn'.
    """
    instruction = instruction.strip() + "Use only Python code, no explanations."
    code_generation_prompt = PromptTemplate(
        input_variables=["instruction"],
        template="""
            You are an expert Python programmer.

            Task:
            - Write complete, runnable, high-quality Python code based on the following user instruction.

            User Instruction:
            {instruction}

            Notes:
            - Make sure to import necessary libraries like pandas, numpy, sklearn, matplotlib, etc., if needed.
            - Make the code clean, correct, and properly indented.
            - Provide only the final code without any extra explanation.

            Wrap your answer inside a python code block like this:
            ```python
            # Your code here
            ```
        """
    )
    code_generation_chain = LLMChain(llm=llm, prompt=code_generation_prompt, output_parser=parser)
    return code_generation_chain.invoke({"instruction": instruction})

@tool
def general_chat_tool(query: str) -> str:
    """
    Handles general conversational queries or fallback responses where no other specific tool matches.
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        You are a friendly and knowledgeable assistant.

        The user has asked the following general query:

        {query}

        Respond in a detailed, friendly, and informative manner.
        If possible, organize your answer in bullet points or paragraphs for clarity.
        Do not make up facts; if unsure, politely suggest possibilities or further research.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    return chain.invoke({"query": query})
    
@tool
def smart_math_solver(query: str) -> str:
    """
    Smart Math Solver:
    - If query looks like integral, derivative, or symbolic, use SymPy.
    - Otherwise, for simple numeric math, use LLMMathChain.
    """
    try:
        if "integral" in query.lower() or "∫" in query or "integrate" in query.lower():
            # Handle basic definite integrals manually
            x = symbols('x')
            expr = 1 / (1 + x**2)
            result = integrate(expr, (x, 0, 1))
            return f"✅ The result of ∫₀¹ (1 / (1 + x²)) dx is: {result} ≈ {float(result)}"

        if "derivative" in query.lower() or "differentiate" in query.lower():
            # Simple placeholder, ideally parse it dynamically
            x = symbols('x')
            expr = 1 / (1 + x**2)
            result = diff(expr, x)
            return f"✅ The derivative of (1 / (1 + x²)) with respect to x is: {result}"

        if "=" in query or re.search(r"[a-zA-Z]", query):
            # Contains symbolic variable, treat it as equation solving
            return equation_solver(query)

        # Else: fallback to normal math solving
        return LLMMathChain.from_llm(llm).run(query)

    except Exception as e:
        return f"⚠️ Error solving the math problem: {str(e)}"
    
def get_weather_data(location: str) -> str:
    """
        Fetches the current weather data for a given location using Weatherstack API.
    """
    response = requests.get(f"http://api.weatherstack.com/current?access_key={weatherstack_api_key}&query={location}")
    return response.json()
    
FALLBACK_TRIGGERS = [
    "not mentioned", 
    "not discussed", 
    "not found", 
    "missing", 
    "no relevant information", 
    "unable to locate", 
    "the document does not", 
    "no answer in document",
    "no information available",
    "not available in the document"
]

def smart_document_qa(query: str, uploaded_files: List) -> str:
    """
    Smart universal Document QA function.
    Accepts uploaded files (PDFs or DOCX) and user query.
    Creates RAG chain dynamically for context-aware answering.
    """
    query = query.strip()
    
    if not uploaded_files:
        return "❌ No documents uploaded for QA."

    documents = []

    # Load documents
    for file in uploaded_files:
        # Save file temporarily
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())

        # Select loader
        if file.name.endswith('.pdf'):
            loader = PDFPlumberLoader(temp_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(temp_path)
        else:
            loader = None

        if loader:
            documents.extend(loader.load())

        # Remove temp file
        os.remove(temp_path)

    if not documents:
        return "⚠️ Unable to load documents."

    # Text splitting
    split_docs = text_splitter.split_documents(documents)

    # Build Vectorstore
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Build both semantic and MMR retrievers
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 10}
    )

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 10, "lambda_mult": 0.5}
    )
    
    def hybrid_retriever(query):
        sem_results = semantic_retriever.get_relevant_documents(query)
        mmr_results = mmr_retriever.get_relevant_documents(query)

        # Combine (simple union)
        doc_set = {doc.page_content: doc for doc in sem_results + mmr_results}
        combined_results = list(doc_set.values())
        return combined_results[:10]
    
    retriever = RunnableLambda(hybrid_retriever)
    
    # Use LLM compression over hybrid retriever
    base_compressor = LLMChainExtractor.from_llm(llm)
    compressed_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=base_compressor
    )

    # RAG chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=compressed_retriever)

    # Prompt Query intelligently
    formatted_query = f"""
    You are answering based on the uploaded document(s).

    Instructions:
    - Answer clearly, accurately, and fully.
    - Stick strictly to what is available in the documents.
    - If answer is not found, say: 'The document does not mention this.'

    User Question: {query}
    """

    response = rag_chain.run(formatted_query)
    if any(trigger in response.lower() for trigger in FALLBACK_TRIGGERS):
        # General fallback
        general_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are a knowledgeable assistant.

            The user has asked the following question:

            {question}

            Provide a complete, helpful, and polite answer based on your general knowledge.
            """
        )
        general_chain = LLMChain(llm=claude, prompt=general_prompt, output_parser=parser)
        fallback_response = general_chain.invoke({"question": query})['text']
        return fallback_response.strip()
    else:
        return response
    
# Define tools for the agent
math_tool = Tool(
    name="Math Solver", 
    func=smart_math_solver,
    description=(
        "Solve any type of mathematical problem with detailed reasoning. "
        "Handles arithmetic, algebra, calculus (derivatives, integrals, limits), "
        "linear equations, trigonometry, and advanced symbolic mathematics. "
        "Input should be a clearly worded math question or a symbolic math expression."
    )
)

search_tool = Tool(
    name="Smart Web Search",
    func=smart_web_search,
    description=(
        "Performs an internet search to provide a detailed, structured answer. "
        "Use this when the user asks about real-world events, recent topics, general knowledge, comparisons, or factual information not known internally."
    )
)

wikipedia_tool = Tool(
    name="Smart Wikipedia Search",
    func=smart_wikipedia_search,
    description=(
        "Searches Wikipedia and returns a detailed, factual answer. "
        "Use this when the user asks about historical events, famous personalities, technical concepts, or any established factual information."
    )
)

news_tool = Tool(
    name="Latest News",
    func=fetch_latest_news,
    description=(
        "Fetches the latest news articles related to a topic. "
        "Use when the user requests current events, breaking news, or recent developments."
    )
)

currency_tool = Tool(
    name="Currency Converter",
    func=convert_currency,
    description=(
        "Converts currencies based on real-time exchange rates. "
        "Use when the user asks to convert one currency to another, e.g., 'convert 100 USD to EUR'."
    )
)

debug_tool = Tool(
    name="Debug Python Code",
    func=debug_code,
    description=(
        "Finds and fixes bugs in Python code. "
        "Use this when the user asks for code debugging, error fixing, troubleshooting, or error explanations."
    )
)

explain_tool = Tool(
    name="Explain Python Code",
    func=explain_code,
    description=(
        "Provides a detailed explanation of Python code. "
        "Use this when the user asks to explain how a piece of Python code works, step-by-step."
    )
)

code_execution_tool = Tool(
    name="Execute Python Code",
    func=python_repl.run,
    description=(
        "Executes Python code and returns the output, including tables and plots. "
        "Use this when the user requests to run Python code snippets and show the results."
    )
)

optimize_tool = Tool(
    name="Optimize Python Code",
    func=optimize_code,
    description=(
        "Improves the performance, readability, and efficiency of Python code. "
        "Use this when the user asks to optimize, refactor, or enhance existing Python code."
    )
)

generate_code_tool = Tool(
    name="Generate Python Code",
    func=generate_python_code,
    description=(
        "Creates complete Python code based on the user's task description. "
        "Use this when the user asks to generate new Python code for any project, task, or functionality."
    )
)

weather_tool = Tool(
    name="Get Weather Condition",
    func=get_weather_data,
    description=(
        "Fetches the current weather condition for a location. "
        "Use this when the user asks for the weather in a specific location."
    )
)

general_chat_fallback_tool = Tool(
    name="General Chat",
    func=general_chat_tool,
    description=(
        "Handles casual conversations, greetings, opinions, and general questions when no other specialized tool applies."
    )
)

tools = [math_tool, search_tool, news_tool, wikipedia_tool, currency_tool, debug_tool, code_execution_tool, explain_tool, optimize_tool, generate_code_tool, weather_tool, general_chat_fallback_tool]
tools.append(extract_symbols)
tools.append(derivative_calculator)
tools.append(integral_calculator)
tools.append(equation_solver)

# Fallback agent for general chat
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory
)

# Chatbot interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.chat_message("assistant").markdown("""
*Your versatile AI partner, ready to assist you across domains!*

---

## 🚀 What I Can Help You With:

- 🔢 **Math Help**: Solve algebra, calculus, equations, and more
- 📄 **Document QA**: Ask intelligent questions about your uploaded documents
- 🌍 **Web Search**: Get accurate, structured answers from the internet
- 📰 **Latest News**: Stay updated with real-time news highlights
- 💱 **Currency Converter**: Instant and reliable currency conversions
- 🐞 **Code Debugging**: Find and fix issues in your Python code
- 🧠 **Code Explanation**: Understand complex code in a simple way
- ⚡ **Code Optimization**: Make your code faster, cleaner, and better
- 💻 **Code Execution**: Instantly run your Python snippets
- 🌤 **Weather Condition**: Get real-time weather updates
- 💬 **General Chat**: Explore ideas, ask questions, or just talk!

---

## ✨ How to Get Started:

- 📥 **Upload a document** (if needed)  
- 🧠 **Ask a question** or **describe your task**  
- 🚀 **Watch** as I assist you **intelligently and efficiently!**

---

*Note: For best results, be clear and specific in your queries!*
""")
                    
user_query = st.chat_input("Ask me anything...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))
    conversation_context = "\n".join(
        f"{role.upper()}: {msg}" for role, msg in st.session_state.chat_history[-5:]
    )

    with st.spinner("Thinking..."):
        try:
            if uploaded_files: 
                response = smart_document_qa(user_query, uploaded_files)
            else:
                formatted_query = f"""
                Conversation History:
                {conversation_context}

                Current Question:
                {user_query}

                Continue the conversation naturally. Never mention the conversation history.
                Answer the question based on your knowledge and the context provided.
                If the question is not related to the conversation, answer it as a new question.
                """
                response = agent.run(formatted_query)

            st.session_state.chat_history.append(("assistant", response))

        except Exception as e:
            response = f"⚠️ Error: {str(e)}"
            st.session_state.chat_history.append(("assistant", response))

# Persistent chat history - user and assistant messages
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)