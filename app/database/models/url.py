from sqlalchemy import (
    Column,
    Integer,
    Text,
    ForeignKey,
    UniqueConstraint,
    DateTime,
    Float,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.db_sqlalchemy import Base


# Modelo de la base de datos
class URL(Base):
    __tablename__ = "urls"

    id_url = Column(Integer, primary_key=True, autoincrement=True)
    id_domain = Column(Integer, ForeignKey("domains.id_domain"), nullable=False)
    relative_url = Column(Text, nullable=False)
    absolute_url = Column(Text, unique=True, nullable=False)
    html_content = Column(Text)
    created_at = Column(DateTime, default=datetime.now())
    modified_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())

    domain = relationship("Domain", back_populates="urls")

    # Relación inversa
    urls_sexist_content = relationship(
        "URLSexistContent", back_populates="url", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("id_domain", "relative_url"),)


class URLSexistContent(Base):
    __tablename__ = "urls_sexist_content"

    id_url_sexist_content = Column(Integer, primary_key=True, autoincrement=True)
    id_url = Column(Integer, ForeignKey("urls.id_url"), nullable=False)
    content = Column(Text, nullable=False)
    sexist = Column(Integer, nullable=False)
    score_sexist = Column(Float, nullable=False)
    score_non_sexist = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now())
    modified_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())

    url = relationship("URL", back_populates="urls_sexist_content")

    __table_args__ = (UniqueConstraint("id_url", "content"),)
