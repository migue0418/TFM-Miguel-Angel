from sqlalchemy import Column, Integer, Text, ForeignKey, UniqueConstraint, DateTime
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

    __table_args__ = (UniqueConstraint("id_domain", "relative_url"),)
