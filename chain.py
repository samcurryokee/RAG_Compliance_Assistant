import os
import re
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

COLLECTION_NAME = "kenyan_constitution"
TOP_K = 5

# Must be the same embedding model used in ingest.py
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# No pre_delete_collection here: that would wipe your ingested data
store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=os.environ["DATABASE_URL"],
    use_jsonb=True,
)
retriever = store.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

prompt = ChatPromptTemplate.from_template("""You are a legal research assistant for the Constitution of Kenya.
Answer the question using ONLY the context below.

Rules:
1. Say "I could not find this in the provided documents." ONLY if none of the passages
   is relevant to the question. If any passage is related, answer from it (see rule 5).
2. Cite every claim as [filename, p.N], copying the label shown above the passage EXACTLY.
   Example: [kenyan_constitution.pdf, p.25]
3. N is always the PAGE number from the label. Never use an Article number as a page number.
4. Only cite pages whose labels appear in the context below.
5. If the passages address the topic only indirectly, do NOT refuse. Say what the text
   does not state, then explain what the relevant passage does say and cite it.
   Example pattern: "The provided text does not mention X by name. However, [provision]
   says ... [citation]."
6. For yes/no questions, answer "yes" or "no" only if the passages directly answer it.
   Otherwise use the pattern in rule 5.
7. Do not conclude beyond what the passages state. If related provisions such as
   exceptions or emergency rules might exist but were not provided, say so.
8. Do not add qualifiers such as "expressly" or "explicitly" unless those words
   appear in the passage.
9. Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:""")

chain = prompt | llm | StrOutputParser()


def page_label(doc):
    # PyPDF pages are 0-indexed; +1 so it matches the printed PDF page
    return int(doc.metadata.get("page", 0)) + 1


def source_label(doc):
    return os.path.basename(doc.metadata.get("source", "unknown"))


def format_docs(docs):
    return "\n\n".join(
        f"[{source_label(d)}, p.{page_label(d)}]\n{d.page_content}" for d in docs
    )


def check_citations(answer, docs):
    """Return the set of cited pages that were NOT among the retrieved pages."""
    retrieved = {page_label(d) for d in docs}
    cited = {int(n) for n in re.findall(r"p\.\s*(\d+)", answer)}
    return sorted(cited - retrieved)


def ask(question):
    docs = retriever.invoke(question)
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    sources = sorted({(source_label(d), page_label(d)) for d in docs})
    bad_pages = check_citations(answer, docs)
    return {"answer": answer, "sources": sources, "docs": docs, "bad_citations": bad_pages}


if __name__ == "__main__":
    print("Kenya Constitution assistant. Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        result = ask(q)
        print(f"\n{result['answer']}\n")
        print("Sources retrieved:")
        for name, page in result["sources"]:
            print(f"  - {name}, p.{page}")
        if result["bad_citations"]:
            pages = ", ".join(f"p.{p}" for p in result["bad_citations"])
            print(f"\n WARNING: answer cites pages that were not retrieved: {pages}")
        print()