import pandas as pd
import re
from fastapi import APIRouter, Depends, HTTPException
from urllib.parse import urlparse

from sqlalchemy.orm import Session
from app.core.config import files_path
from app.database.db_service import DB
from app.database.db_sqlalchemy import get_db
from app.utils.web_crawling import get_sitemaps_from_domain, get_all_urls_and_sitemaps

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
        domain_instance = db_manager.save_domain(domain_url=netloc,
                                                 absolute_domain=absolute_domain)

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
