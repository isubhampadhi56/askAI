from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source = Column(String)
    content = Column(Text)
    embedding = Column(Vector(1536)) # match embed dim
class Users(Base):
    __tablename__ = "users"
    id = Column(Integer,primary_key=True)
    name = Column(String)
    username = Column(String, unique=True)
    password = Column(String)


