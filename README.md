# 🚀 RAG Agent with Cascade Retrieval

This repository implements a **Retrieval Augmented Generation (RAG) agent** that utilizes a **cascade retrieval approach**. The cascade retriever first applies **sparse BM25 retrieval** to quickly narrow down the corpus to a set of candidate documents and then **re-ranks these candidates using dense retrieval via FAISS embeddings**. This dual-stage process offers improved precision over traditional RAG systems that rely solely on a single dense retrieval step.

---

## 📥 Cloning the Repository

Clone the repository using Git:

```bash
git clone https://github.com/yourusername/rag-agent.git
Then, navigate into the project directory:

bash
Copy
Edit
cd rag-agent
⚙️ Setup
1️⃣ Install Dependencies
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Environment Variables
Create a .env file in the root directory and add your OpenAI API key:

ini
Copy
Edit
OPENAI_API_KEY=your-api-key
🔍 How Cascade Retrieval Works
Step 1: Sparse Retrieval (BM25)
The agent uses BM25 to perform an initial, fast retrieval over the full corpus.
It returns a broader set of candidate documents, efficiently reducing the search space.
Step 2: Dense Retrieval (FAISS)
The BM25 candidate documents are re-ranked using dense retrieval.
OpenAI embeddings compute semantic similarities to the query.
FAISS efficiently retrieves the top relevant documents from the candidates.
🔹 This method leverages both sparse and dense techniques, ensuring a balance between speed and accuracy.
For more details, refer to the implementation in the source code (see rag_agent.py).

🚀 Running the Agent
Execute the following command to run the RAG agent:

bash
Copy
Edit
python rag_agent.py
The agent will: ✅ Load documents from the specified directory.
✅ Apply cascade retrieval to extract relevant context based on your query.
✅ Use an LLM (e.g., GPT-4) to generate a response.

📜 License
[Include your license information here.]

🤝 Contributing
Contributions are welcome! Please follow the repository guidelines for submitting updates and improvements.

