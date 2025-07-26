import pandas as pd
import re
from fastapi import APIRouter, Depends, HTTPException
from urllib.parse import urlparse

from sqlalchemy.orm import Session
from app.core.config import files_path
from app.database.crud.domain import (
    get_domains,
    save_domain,
    delete_domain,
    update_domain,
)
from app.database.crud.url import get_urls, save_url, delete_url, update_url
from app.database.db_service import DB
from app.database.db_sqlalchemy import get_db
from app.schemas.inputs import TextInput
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


@router.post("/sitemap/get-urls")
def save_english_urls_from_sitemap_domain(domain: str, db: Session = Depends(get_db)):
    """
    Get a list of the English urls from a domain sitemap and save it in the database
    """
    try:
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
        all_urls, all_sitemaps = get_all_urls_and_sitemaps(sitemaps_urls)

        # Parse the set urls into a list
        all_urls = list(all_urls)

        # Iterate the URLs and add them to the database
        for url in all_urls:
            # Add it to the database if not already there
            db_manager.save_url(domain_instance, url)

        # Save the urls in a csv file using pandas
        df = pd.DataFrame(all_urls, columns=["url"])

        # Save the csv file in the correct folder
        output_csv_path = (
            files_path
            / "domain_urls"
            / f"{re.sub(r"^https?://", "", domain).replace('.', '_')}_english_urls.csv"
        )
        df.to_csv(output_csv_path, index=False)

        return {
            "message": (
                f"Found {len(all_urls)} English urls from the domain ",
                f"{domain} in {len(all_sitemaps)} sitemaps",
            )
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


@router.get("/url/check-sexism")
def check_sexism_in_url(
    url: str, filter_tag: str = None, db: Session = Depends(get_db)
):
    """
    Get a list of the English urls from a domain sitemap saved in the database
    """
    try:
        # Instancia la base de datos
        db_manager = DB(db)

        # Parseamos la url para asegurarnos de que es válida
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if not domain:
            raise HTTPException(status_code=400, detail="Invalid URL provided")

        # Comprobamos el dominio de la URL y lo intentamos guardar
        domain = db_manager.save_domain(
            domain_url=domain, absolute_domain=f"{parsed_url.scheme}://{domain}"
        )

        # Obtenemos el contenido
        content = get_url_html_content(url)
        if not content:
            raise HTTPException(
                status_code=404, detail="Content not found or inaccessible"
            )

        # Comprobamos si la URL ya está en la base de datos
        url_instance = db_manager.save_url(domain, url, content)

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
            db_manager.save_url_sexist_content(url_instance, result)

        return results

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
