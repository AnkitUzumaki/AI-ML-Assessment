# Web Content Q&A Tool 🌐❓

A Streamlit application that extracts text from URLs and answers questions using only the provided content.

## Features ✨
- Extract main content from multiple URLs
- Semantic search using embeddings
- Context-aware question answering
- Clean Streamlit UI

## Installation 💻
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Navigate to project directory
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt
```

## Usage 🛠️
1. Run the application:
```bash
streamlit run app.py
```
2. Enter one or more URLs
3. Ask questions about the content

## Dependencies 📦
- Python 3.8+
- Streamlit
- BeautifulSoup4
- LangChain
- FAISS
- Sentence Transformers

## Project Structure 📂
```
project/
├── app.py             
├── requirements.txt   
├── README.md          
```

## How It Works 🔍
1. **Content Extraction**: Uses BeautifulSoup to scrape main text
2. **Vector Storage**: Creates embeddings using Sentence Transformers
3. **Question Answering**: Retrieves relevant context using FAISS
4. **Response Generation**: Answers using only the provided content

## Contributing 🤝
Pull requests welcome! For major changes, please open an issue first.
