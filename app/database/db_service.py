from typing import Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from app.database.models.domain import Domain
from app.database.models.url import URL, URLSexistContent


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

    def save_url(self, domain: Domain, url: str, html_content: Optional[str] = None):
        url_instance = (
            self.db.query(URL)
            .filter_by(id_domain=domain.id_domain, absolute_url=url)
            .first()
        )

        if not url_instance:
            relative_url = url.replace(domain.absolute_url, "", 1)
            url_instance = URL(
                id_domain=domain.id_domain,
                absolute_url=url,
                relative_url=relative_url,
                html_content=html_content,  # Assuming html_content is optional and can be set later
            )
            self.db.add(url_instance)
            self.db.commit()
            self.db.refresh(url_instance)

        # Si no tiene contenido HTML, lo actualizamos
        if url_instance.html_content is None and html_content is not None:
            url_instance.html_content = html_content
            url_instance.modified_at = datetime.now()
            self.db.commit()
            self.db.refresh(url_instance)

        return url_instance

    def get_all_urls(
        self, domain_id: Optional[int] = None, domain_url: Optional[str] = None
    ):
        if domain_id is not None:
            return self.db.query(URL).filter_by(id_domain=domain_id).all()
        elif domain_url is not None:
            domain = self.db.query(Domain).filter_by(domain_url=domain_url).first()
            if domain:
                return self.db.query(URL).filter_by(id_domain=domain.id_domain).all()

    def get_urls_not_checked(
        self, domain_id: Optional[int] = None, domain_url: Optional[str] = None
    ):
        if domain_id is not None:
            return (
                self.db.query(URL)
                .filter_by(id_domain=domain_id)
                .filter(URL.urls_sexist_content is None)
                .all()
            )
        elif domain_url is not None:
            domain = self.db.query(Domain).filter_by(domain_url=domain_url).first()
            if domain:
                return (
                    self.db.query(URL)
                    .filter_by(id_domain=domain.id_domain)
                    .filter(URL.urls_sexist_content is None)
                    .all()
                )
        return []

    def save_url_sexist_content(self, url: URL, sexism_pred: dict[str, Any]):
        row = (
            self.db.query(URLSexistContent)
            .filter_by(id_url=url.id_url, content=sexism_pred["text"])
            .first()
        )
        if not row:
            row = URLSexistContent(
                id_url=url.id_url,
                content=sexism_pred["text"],
                sexist=1 if sexism_pred["pred"] == "sexist" else 0,
                score_sexist=sexism_pred["score_sexist"],
                score_non_sexist=sexism_pred["score_not_sexist"],
            )
            self.db.add(row)
            self.db.commit()
            self.db.refresh(row)
        return row
