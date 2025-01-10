import pandas as pd
import re
from fastapi import APIRouter, HTTPException
from app.core.config import files_path
from app.utils.web_crawling import get_sitemaps_from_domain, get_all_urls_and_sitemaps

router = APIRouter(
    prefix="/web-crawling",
    tags=["Web Crawling"],
)


@router.post("/sitemap/get-urls")
def save_english_urls_from_sitemap_domain(domain: str):
    """
    Get a list of the English urls from a domain sitemap and save it in the database
    """
    try:
        # Eliminar la barra derecha
        domain = domain.rstrip("/")

        # Get the sitemap list of the domain
        sitemaps_urls = get_sitemaps_from_domain(domain)

        # Get the English urls from the sitemaps
        all_urls, all_sitemaps = get_all_urls_and_sitemaps(sitemaps_urls)

        # Parse the set urls into a list
        all_urls = list(all_urls)

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
