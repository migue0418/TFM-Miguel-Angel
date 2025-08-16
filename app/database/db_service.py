from typing import Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, select, desc
from datetime import date
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

    def get_overview(self, from_date: date | None = None, to_date: date | None = None):
        """
        Devuelve (total_urls, total_sentences, global_sexism_percentage)
        """
        # Filtro de rango de fechas (se asume URL.created_at existe)
        url_filter = []
        if from_date:
            url_filter.append(URL.created_at >= from_date)
        if to_date:
            url_filter.append(URL.created_at <= to_date)

        # Query de URLs
        total_urls = self.db.scalar(
            select(func.count()).select_from(URL).where(*url_filter)
        )

        # Sentencias unidas a URLs dentro del rango
        sent_query = select(URLSexistContent).join(
            URL, URLSexistContent.id_url == URL.id_url
        )
        if url_filter:
            sent_query = sent_query.where(*url_filter)

        total_sentences = self.db.scalar(
            select(func.count()).select_from(sent_query.subquery())
        )

        sexist_sentences_q = (
            select(func.count())
            .select_from(URLSexistContent)
            .join(URL, URLSexistContent.id_url == URL.id_url)
            .where(*url_filter, URLSexistContent.sexist == 1)
        )
        sexist_sentences = self.db.scalar(sexist_sentences_q)

        percentage = (
            (sexist_sentences / total_sentences) * 100 if total_sentences else 0.0
        )

        return (
            total_urls or 0,
            total_sentences or 0,
            sexist_sentences,
            round(percentage, 2),
        )

    def get_top_sentences(self, limit: int = 5):
        """
        Devuelve una lista de dicts [{text, score_sexist, url_id?, created_at?}, ...]
        ordenada por score_sexist DESC.
        """
        stmt = (
            select(
                URLSexistContent.content,
                URLSexistContent.score_sexist,
            )
            .where(URLSexistContent.sexist == 1)
            .order_by(desc(URLSexistContent.score_sexist))
            .limit(limit)
        )

        rows = self.db.execute(stmt).all()

        return [
            {"text": r.content, "score_sexist": float(r.score_sexist)} for r in rows
        ]

    def get_severity_histogram(self, bins: int = 5):
        """
        Devuelve lista [{range: '0-0.2', count: 123}, ...] con 'bins' intervalos
        equiespaciados en [0,1].
        """
        # floor(score * bins) → índice de bin (0 … bins-1)
        bin_index = func.floor(URLSexistContent.score_sexist * bins)

        stmt = select(bin_index.label("bin"), func.count().label("count")).group_by(
            "bin"
        )

        counts = {int(b): c for b, c in self.db.execute(stmt).all()}

        # Formateo uniforme de los bins (incluso vacíos)
        hist = []
        step = 1 / bins
        for i in range(bins):
            low = round(i * step, 2)
            high = round((i + 1) * step, 2)
            label = f"{low:.1f}-{high:.1f}"
            hist.append({"range": label, "count": counts.get(i, 0)})

        return hist
