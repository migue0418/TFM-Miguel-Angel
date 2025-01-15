from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.db_sqlalchemy import Base

# Modelo de la base de datos
class Domain(Base):
    __tablename__ = "domains"

    id_domain = Column(Integer, primary_key=True, autoincrement=True)
    domain_url = Column(String, unique=True, nullable=False)
    absolute_url = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now())

    urls = relationship("URL", back_populates="domain")
