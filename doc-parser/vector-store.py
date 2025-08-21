from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from PyPDF2 import PdfReader
from markdown import markdown
from bs4 import BeautifulSoup
from db.schemas import Base, Document
from embedder import Embedder

class VectorStore:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.embedder = Embedder()

    def add_texts(self, texts, sources):
        vecs = self.embedder.embed(texts)
        with self.Session() as s:
            for t, src, v in zip(texts, sources, vecs):
                doc = Document(source=src, content=t, embedding=v)
                s.add(doc)
            s.commit()

    def search(self, query: str, k: int = 4):
        q_vec = self.embedder.embed([query])[0]
        with self.Session() as s:
            rows = s.execute(
                text("""
                SELECT id, source, content, 1 - (embedding <=> :q) AS score
                FROM documents
                ORDER BY embedding <=> :q
                LIMIT :k
                """),
                {"q": q_vec, "k": k}
            ).mappings().all()
            return [dict(r) for r in rows]

    def _read_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_md(self, path: str) -> str:
        html = markdown(open(path, "r", encoding="utf-8").read())
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ")

    def _read_txt(self, path: str) -> str:
        return open(path, "r", encoding="utf-8").read()

    READERS = {".pdf": _read_pdf, ".md": _read_md, ".txt": _read_txt}
