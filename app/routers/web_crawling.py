from datetime import date
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from urllib.parse import urlparse
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.database.crud.domain import (
    get_domains,
    save_domain,
    delete_domain,
    update_domain,
)
from app.database.crud.url import get_urls, save_url, delete_url, update_url
from app.database.db_service import DB
from app.database.db_sqlalchemy import get_db, SessionLocal
from app.database.models.url import URLSexistContent, URL
from app.schemas.inputs import DomainAnalyzerInput, TextInput, URLCheckerInput
from app.schemas.web import EditDomain, NewDomain, EditURL, NewURL
from app.utils.sexism_classification import predict_sexism_text
from app.utils.web_crawling import (
    get_sitemaps_from_domain,
    get_all_urls_and_sitemaps,
    get_url_html_content,
    get_url_texts_content,
    split_text_into_sentences,
)

router = APIRouter(
    prefix="/web-crawling",
    tags=["Web Crawling"],
)


# Wrapper para ejecutar en background con su propia sesión
def run_check_urls_not_checked(absolute_domain: str) -> None:
    db_bg = SessionLocal()
    try:
        # Llama a tu función real que procesa las URLs pendientes
        check_urls_not_checked(domain=absolute_domain, db=db_bg)
    except Exception as e:
        # Aquí puedes loguear el error si quieres
        print(f"[BG] Error en check_urls_not_checked({absolute_domain}): {e}")
    finally:
        db_bg.close()


@router.post("/sitemap/get-urls")
def save_english_urls_from_sitemap_domain(
    form_data: DomainAnalyzerInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Get a list of the English urls from a domain sitemap and save it in the database
    """
    try:
        domain = form_data.domain.strip()
        if not domain:
            raise HTTPException(status_code=400, detail="Domain cannot be empty")
        # Instancia la base de datos
        db_manager = DB(db)

        # Eliminar la barra derecha
        domain = domain.rstrip("/")

        # Parseamos la url del dominio con urlparse
        parsed_domain = urlparse(domain)
        scheme = parsed_domain.scheme or "https"
        netloc = parsed_domain.netloc or domain
        absolute_domain = f"{scheme}://{netloc}"

        # Add it to the database if not already there
        domain_instance = db_manager.save_domain(
            domain_url=netloc, absolute_domain=absolute_domain
        )

        # Get the sitemap list of the domain
        sitemaps_urls = get_sitemaps_from_domain(absolute_domain)

        # Get the English urls from the sitemaps
        sitemaps, all_urls = get_all_urls_and_sitemaps(sitemaps_urls)

        # Parse the set urls into a list
        all_urls = list(all_urls)

        # Iterate the URLs and add them to the database
        for url in all_urls:
            # Add it to the database if not already there
            db_manager.save_url(domain_instance, url)

        # Mandamos a background la tarea de ejecutar los sitemaps
        background_tasks.add_task(run_check_urls_not_checked, absolute_domain)

        return {
            "id_domain": domain_instance.id_domain,
            "domain": domain_instance.domain_url,
            "sitemaps": sitemaps,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/domains/urls")
def get_domain_urls_in_db(domain: str, db: Session = Depends(get_db)):
    """
    Get a list of the English urls from a domain sitemap saved in the database
    """
    try:
        # Instancia la base de datos
        db_manager = DB(db)

        # Eliminar la barra derecha
        domain = domain.rstrip("/")

        # Parseamos la url del dominio con urlparse
        parsed_domain = urlparse(domain)
        netloc = parsed_domain.netloc or domain

        # Add it to the database if not already there
        domain_urls = db_manager.get_all_urls(domain_url=netloc)

        return domain_urls

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/domains/check-urls-not-checked")
def check_urls_not_checked(
    domain: str, filter_tag: str = None, db: Session = Depends(get_db)
):
    """
    Get the urls from a domain that have not been checked for sexism
    """
    try:
        # Instancia la base de datos
        db_manager = DB(db)

        # Eliminar la barra derecha
        domain = domain.rstrip("/")

        # Parseamos la url del dominio con urlparse
        parsed_domain = urlparse(domain)
        netloc = parsed_domain.netloc or domain

        # Get all the urls from the domain
        urls_not_checked = db_manager.get_urls_not_checked(domain_url=netloc)

        for url in urls_not_checked:
            # If the url has no html content, we fetch it
            if not url.html_content:
                content = get_url_html_content(url.absolute_url)
                if content:
                    # Update the url with the fetched content
                    url = db_manager.save_url(url.domain, url.absolute_url, content)

                # Obtenemos el texto de la URL usando BeautifulSoup
                texts = get_url_texts_content(html_text=content, filter_tag=filter_tag)
                if not texts:
                    raise HTTPException(
                        status_code=404, detail="No text content found in the URL"
                    )

                # Verificamos el sexismo en los textos
                results = predict_sexism_text(texts)

                for result in results:
                    # Guardamos el contenido sexista en la base de datos
                    db_manager.save_url_sexist_content(url, result)

        return urls_not_checked

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/url/check-sexism")
def check_sexism_in_url(form_data: URLCheckerInput, db: Session = Depends(get_db)):
    """
    Check sexism in a URL and save the results in the database.
    If the URL is not valid or the content is not accessible, it raises an HTTPException.
    If the URL has already been checked, it returns the existing results.
    """
    try:
        # Instancia la base de datos
        db_manager = DB(db)

        # Parseamos la url para asegurarnos de que es válida
        parsed_url = urlparse(form_data.url)
        domain = parsed_url.netloc
        if not domain:
            raise HTTPException(status_code=400, detail="Invalid URL provided")

        # Comprobamos el dominio de la URL y lo intentamos guardar
        domain = db_manager.save_domain(
            domain_url=domain, absolute_domain=f"{parsed_url.scheme}://{domain}"
        )

        # Obtenemos el contenido
        content = get_url_html_content(form_data.url)
        if not content:
            raise HTTPException(
                status_code=404, detail="Content not found or inaccessible"
            )

        # Comprobamos si la URL ya está en la base de datos
        url_instance = db_manager.save_url(domain, form_data.url, content)

        # Obtenemos el texto de la URL usando BeautifulSoup
        texts = get_url_texts_content(
            html_text=content, filter_tag=form_data.filter_tag
        )
        if not texts:
            raise HTTPException(
                status_code=404, detail="No text content found in the URL"
            )

        # Verificamos el sexismo en los textos
        results = predict_sexism_text(texts)

        for result in results:
            # Guardamos el contenido sexista en la base de datos
            db_manager.save_url_sexist_content(url_instance, result)

        total_texts = len(texts)
        sexist_texts = sum(1 for r in results if r["pred"] == "sexist")
        sexism_percentage = (sexist_texts / total_texts) * 100 if total_texts else 0

        response = {
            "global": {
                "is_sexist": sexist_texts >= (total_texts - sexist_texts),
                "sexism_percentage": sexism_percentage,
                "total_texts": total_texts,
            },
            "texts": results,
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/text/check-sexism")
def check_sexism_in_text(text: TextInput):
    """
    Get a list of the English urls from a domain sitemap saved in the database
    """
    try:
        text = text.text.strip()

        # Comprobamos si el texto es válido
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Invalid text provided")

        # Obtenemos las frases del texto
        texts = split_text_into_sentences(text)

        # Verificamos el sexismo en los textos
        results = predict_sexism_text(texts)

        # Obtenemos el porcentaje global de sexismo
        total_texts = len(texts)

        # Calculamos el número de textos sexistas y no sexistas
        sexist_texts = sum(1 for result in results if result["pred"] == "sexist")
        not_sexist_texts = total_texts - sexist_texts

        # Calculamos el porcentaje de sexismo
        sexism_percentage = (sexist_texts / total_texts) * 100 if total_texts > 0 else 0

        # Devolvemos un JSON con los resultados globales y por frases
        results = {
            "global": {
                "is_sexist": sexist_texts >= not_sexist_texts,
                "sexism_percentage": sexism_percentage,
                "total_texts": total_texts,
            },
            "texts": results,
        }

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/domains")
async def _get_domains(domain: str = None):
    """
    Get all domains or filter by a specific domain.
    If domain is provided, it returns only that domain; otherwise, it returns all domains.
    """
    try:
        return await get_domains(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching domains: {str(e)}")


@router.get("/domain/{id_domain}")
async def _get_domain_by_id(id_domain: int):
    """
    Get a domain by its ID.
    """
    try:
        return await get_domains(id_domain=id_domain)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching domain by ID {id_domain}: {str(e)}"
        )


@router.post("/domains")
async def _save_domain(form_data: NewDomain):
    """
    Save a new domain to the database.
    If the domain already exists, it returns the existing domain.
    """
    try:
        return await save_domain(form_data.domain_url, form_data.absolute_url)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving domain {form_data.domain_url}: {str(e)}",
        )


@router.delete("/domain/{id_domain}")
async def _delete_domain(id_domain: int):
    """
    Delete a domain by its ID.
    """
    try:
        return await delete_domain(id_domain)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting domain with ID {id_domain}: {str(e)}",
        )


@router.put("/domain/{id_domain}")
async def _update_domain(form_data: EditDomain):
    """
    Update an existing domain's URL or absolute URL.
    If the domain does not exist, it raises an error.
    """
    try:
        return await update_domain(
            id_domain=form_data.id_domain,
            domain_url=form_data.domain_url,
            absolute_url=form_data.absolute_url,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating domain with ID {form_data.id_domain}: {str(e)}",
        )


@router.get("/urls")
async def _get_urls(
    id_domain: int = None, domain_url: str = None, url: str = None, id_url: int = None
):
    """
    Get URLs based on various filters.
    If id_domain or domain_url is provided, it returns URLs for that domain.
    If url or id_url is provided, it returns a specific URL.
    """
    try:
        return await get_urls(id_domain, domain_url, url, id_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching URLs: {str(e)}")


@router.post("/urls")
async def _save_url(form_data: NewURL):
    """
    Save a new URL to the database.
    If the URL already exists, it returns the existing URL.
    """
    try:
        return await save_url(
            form_data.id_domain,
            form_data.absolute_url,
            form_data.relative_url,
            form_data.html_content,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving URL {form_data.absolute_url}: {str(e)}",
        )


@router.delete("/url/{id_url}")
async def _delete_url(id_url: int):
    """
    Delete a URL by its ID.
    """
    try:
        return await delete_url(id_url)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting URL with ID {id_url}: {str(e)}"
        )


@router.put("/url/{id_url}")
async def _update_url(form_data: EditURL):
    """
    Update an existing URL's absolute URL, relative URL, or HTML content.
    If the URL does not exist, it raises an error.
    """
    try:
        return await update_url(
            form_data.id_url,
            form_data.absolute_url,
            form_data.relative_url,
            form_data.html_content,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating URL with ID {form_data.id_url}: {str(e)}",
        )


@router.get("/domain/{id_dominio}/urls")
def get_urls_by_domain(id_dominio: int, db: Session = Depends(get_db)):
    try:
        # Subquery: media y flag "hay frases sexistas"
        subq = (
            db.query(
                URLSexistContent.id_url,
                func.avg(URLSexistContent.score_sexist).label("score_sexist_global"),
                func.max(URLSexistContent.sexist).label("has_sexist_parts"),  # 0/1
            )
            .group_by(URLSexistContent.id_url)
            .subquery()
        )

        # Join entre urls y subquery
        results = (
            db.query(
                URL.id_url,
                URL.relative_url,
                func.coalesce(subq.c.score_sexist_global, 0).label(
                    "score_sexist_global"
                ),
                func.coalesce(subq.c.has_sexist_parts, 0).label("has_sexist_parts"),
            )
            .outerjoin(subq, URL.id_url == subq.c.id_url)
            .filter(URL.id_domain == id_dominio)
            .all()
        )

        return [
            {
                "id_url": row.id_url,
                "relative_url": row.relative_url,
                # porcentual para el frontend
                "score_sexist_global": round(row.score_sexist_global * 100, 2),
                # True si max(sexist) = 1
                "has_sexist_parts": bool(row.has_sexist_parts),
            }
            for row in results
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/urls/{id_url}/sexism")
def get_url_sexism_analysis(id_url: int, db: Session = Depends(get_db)):
    try:
        # Obtener la URL
        url = db.query(URL).filter(URL.id_url == id_url).first()
        if not url:
            raise HTTPException(status_code=404, detail="URL no encontrada")

        # Obtener los fragmentos analizados
        contents = (
            db.query(URLSexistContent).filter(URLSexistContent.id_url == id_url).all()
        )

        if not contents:
            raise HTTPException(
                status_code=404, detail="No hay análisis asociados a esta URL"
            )

        # Calcular score medio
        total_sentences = len(contents)
        avg_score_sexist = sum(c.score_sexist for c in contents) / total_sentences

        # Construir respuesta
        return {
            "absolute_url": url.absolute_url,
            "global_score": round(avg_score_sexist, 4),
            "total_sentences": total_sentences,
            "texts": [
                {
                    "content": c.content,
                    "sexist": bool(c.sexist),
                    "score_sexist": round(c.score_sexist, 4),
                    "score_non_sexist": round(c.score_non_sexist, 4),
                }
                for c in contents
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/analytics/overview")
def analytics_overview(
    from_date: date | None = None,
    to_date: date | None = None,
    db: Session = Depends(get_db),
):
    """
    Devuelve:
      {
        total_urls: 123,
        total_sentences: 4567,
        global_sexism_percentage: 23.4        # 0-100
      }
    """
    try:
        dbm = DB(db)
        urls, sentences, sexist_sentences, pct = dbm.get_overview(from_date, to_date)
        return {
            "total_urls": urls,
            "total_sentences": sentences,
            "sexist_sentences": sexist_sentences,
            "global_sexism_percentage": pct,
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/top-sexist-sentences")
def top_sexist_sentences(
    limit: int = 5,
    db: Session = Depends(get_db),
):
    """
    Devuelve:
      { sentences: [ { text, score_sexist }, ... ] }
    """
    try:
        dbm = DB(db)
        top = dbm.get_top_sentences(limit=limit)  # ordered by score_sexist DESC
        return {"sentences": top}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/analytics/severity-distribution")
def severity_distribution(
    bins: int = 5,
    db: Session = Depends(get_db),
):
    """
    Devuelve:
      {
        bins: [
          { range: \"0-0.2\", count: 123 },
          { range: \"0.2-0.4\", count: 234 }, ...
        ]
      }
    """
    try:
        dbm = DB(db)
        dist = dbm.get_severity_histogram(bins=bins)
        return {"bins": dist}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
