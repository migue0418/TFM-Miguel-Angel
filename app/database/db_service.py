from sqlalchemy.orm import Session
from datetime import datetime
from app.database.models.domain import Domain
from app.database.models.url import URL

class DB:
    def __init__(self, db: Session):
        self.db = db

    def save_domain(self, domain_url: str, absolute_domain: str):
        domain = self.db.query(Domain).filter_by(domain_url=domain_url).first()
        if not domain:
            domain = Domain(domain_url=domain_url, absolute_url=absolute_domain)
            self.db.add(domain)
            self.db.commit()
            self.db.refresh(domain)
        return domain

    def save_url(self, domain: Domain, url: str):
        url_instance = self.db.query(URL).filter_by(
            id_domain=domain.id_domain,
            absolute_url=url
        ).first()

        if not url_instance:
            relative_url = url.replace(domain.absolute_url, "", 1)
            url_instance = URL(
                id_domain=domain.id_domain,
                absolute_url=url,
                relative_url=relative_url,
            )
            self.db.add(url_instance)
            self.db.commit()
            self.db.refresh(url_instance)
        return url_instance

    def get_all_urls(self, domain_id: int):
        return self.db.query(URL).filter_by(id_domain=domain_id).all()
