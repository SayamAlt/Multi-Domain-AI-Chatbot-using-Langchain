# 🤖 Multi-Domain AI Personal Assistant

![AI Chatbot](https://blog.n8n.io/content/images/size/w1200/2024/06/ai-chatbots-8--1---3-.png)
![AI Agent](https://peritushub.com/wp-content/uploads/2024/06/chatbot.jpg)
![AI Agents Evolution](https://www.gupshup.io/resources/wp-content/uploads/2024/11/DZN-2052-AI-agents-blog-img-1-1024x399.jpg?x97717)

An intelligent, modular, and production-grade **AI Assistant** built using **LangChain**, **OpenAI**, **Anthropic**, **SymPy**, **Python**, and **Streamlit**.

This project enables users to interact with an AI agent that can:

- 📚 **Summarize and QA uploaded documents** (PDFs, DOCX)
- 🧠 **Solve complex mathematical problems** (algebra, calculus, symbolic math)
- 📰 **Fetch the latest news** from real-world sources
- 🌍 **Perform web and Wikipedia searches** with detailed responses
- 🐍 **Debug, explain, optimize, and generate Python code**
- 🖥️ **Execute Python snippets live**
- 💱 **Convert currencies** with real-time rates
- 💬 **Engage in open-ended, context-aware conversations**

---

## 🚀 Features

### 🔥 AI Functionalities

- **Smart Document QA**: Upload any PDF/DOCX, ask questions, get intelligent answers.
- **News Summarization**: Fetches real-world news and summarizes it like a professional report.
- **Web + Wikipedia Search**: Queries DuckDuckGo and Wikipedia, compiles coherent answers.
- **Advanced Math Solver**: Handles symbolic differentiation, integration, equation solving, and standard math.
- **Code Engineering Suite**:
  - Debug Python code (find and fix errors).
  - Explain Python code (step-by-step breakdown).
  - Optimize Python code (performance, best practices).
  - Generate Python code from instructions.
  - Execute Python snippets live.

---

### 🛠️ Technologies Used

- **Python 3.12+**
- **Streamlit** (UI & Deployment)
- **LangChain**
- **OpenAI API (gpt-4o / gpt-3.5-turbo)**
- **Anthropic Claude 3 Sonnet** (backup LLM for reasoning)
- **SymPy** (Math engine)
- **FAISS** (Vector database for document retrieval)
- **DuckDuckGo Search API** (smart web search)
- **NewsAPI / GNews API** (real-time news)
- **ExchangeRate API** (currency conversion)

---

### 🧩 LangChain Concepts Applied

- Tools and Agents (Zero-Shot-ReAct Framework)
- Memory (Context retention across conversation turns)
- Smart retrievals using **Contextual Compression Retriever** + **LLMChain Extractor**
- Custom Runnable chains (`RunnablePassthrough`, `RunnableParallel`)
- Output parsing via **StrOutputParser**
- Error handling and fallback strategies

---

## 📥 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-personal-assistant.git
cd ai-personal-assistant
```

# 2. (Optional) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

# 3. Install all required packages

```bash
pip install -r requirements.txt
```

---

## Required Environment Variables

Create a .env file in the root directory with your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GNEWS_API_KEY=your-newsapi-api-key
EXCHANGE_RATE_API_KEY=your-exchangerate-api-key
```

---

## 🧠 How It Works

<ol>
    <li>**Document Upload**: Upload multiple PDFs or DOCX files to the sidebar.</li>
    <li>**Question Asking**: Ask questions related to the uploaded documents.</li>
    <li>**Tool Routing**: LangChain agent intelligently selects the best tool:</li>
    <li>**SmartMathSolver** for math</li>
    <li>**PythonDebugger** for code</li>
    <li>**SmartWebSearch** for real-world info</li>
    <li>**Fallback Handling**: If the document lacks an answer, the assistant gracefully transitions to general knowledge.</li>
    <li>**Multi-turn Chat**: Context is retained using conversation memory.</li>
</ol>

---

## ✨ Future Enhancements

<ul>
    <li>Add support for image-based document QA (OCR).</li>
    <li>Add user authentication (Streamlit login).</li>
    <li>Deploy a production backend using FastAPI + LangServe.</li>
    <li>Expand toolset to include SQL Query Generation, Visualization Generator, and more.</li>
</ul>

---

## 🙌 Acknowledgements

<ul>
    <li>OpenAI</li>
    <li>LangChain</li>
    <li>Streamlit</li>
    <li>Anthropic</li>
    <li>FAISS</li>
    <li>DuckDuckGo Search API</li>  
    <li>NewsAPI.org</li>
</ul>

---

## 🛡️ License

This project is licensed under the Apache 2.0 License.

---

## 👨‍💻 Author

Name: Sayam Kumar<br>
Email: sayamk565@gmail.com<br>
LinkedIn: https://www.linkedin.com/in/sayam-kumar/

Feel free to contact me through email or LinkedIn in case you have any queries about this project.

> Built with ❤️, Intelligence, and Curiosity.




