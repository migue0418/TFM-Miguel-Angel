from typing import Optional
from app.core.dependencies import db_tfm as db


async def get_urls(
    domain_id: Optional[int] = None,
    domain_url: Optional[str] = None,
    url: Optional[str] = None,
    id_url: Optional[int] = None,
):
    """
    Fetch all URLs or filter by a specific domain ID or URL.
    If domain_id is provided, it returns URLs for that domain; if domain_url is provided, it returns URLs for that domain URL.
    """
    try:
        if domain_id is not None:
            return await db.fetch_all("urls", where={"id_domain": domain_id})
        elif domain_url is not None:
            return await db.fetch_all("urls", where={"domain_url": domain_url})
        elif url is not None:
            return await db.fetch_one("urls", where={"absolute_url": url})
        elif id_url is not None:
            return await db.fetch_one("urls", where={"id_url": id_url})
        else:
            return await db.fetch_all("urls")
    except Exception as e:
        raise RuntimeError(f"Error fetching URLs: {str(e)}")


async def save_url(
    domain_id: int,
    absolute_url: str,
    relative_url: Optional[str] = None,
    html_content: Optional[str] = None,
):
    """
    Save a new URL to the database.
    If the URL already exists, it returns the existing URL.
    """
    try:
        existing_url = await db.fetch_one(
            "urls", where={"absolute_url": absolute_url, "id_domain": domain_id}
        )
        if existing_url:
            return existing_url

        new_url = {
            "id_domain": domain_id,
            "absolute_url": absolute_url,
            "relative_url": relative_url or absolute_url.replace("/", ""),
            "html_content": html_content,
        }
        return await db.insert("urls", new_url, return_id=True)

    except Exception as e:
        raise RuntimeError(f"Error saving URL {absolute_url}: {str(e)}")


async def delete_url(id_url: int):
    """
    Delete a URL by its ID.
    """
    try:
        return await db.delete("urls", where_conditions={"id_url": id_url})
    except Exception as e:
        raise RuntimeError(f"Error deleting URL with ID {id_url}: {str(e)}")


async def update_url(
    id_url: int,
    absolute_url: Optional[str] = None,
    relative_url: Optional[str] = None,
    html_content: Optional[str] = None,
):
    """
    Update an existing URL's absolute URL, relative URL, or HTML content.
    If the URL does not exist, it raises an error.
    """
    try:
        set_fields = {}
        if absolute_url is not None:
            set_fields["absolute_url"] = absolute_url
        if relative_url is not None:
            set_fields["relative_url"] = relative_url
        if html_content is not None:
            set_fields["html_content"] = html_content

        if not set_fields:
            raise ValueError("No fields to update provided.")

        return await db.update("urls", set_fields, where={"id_url": id_url})

    except Exception as e:
        raise RuntimeError(f"Error updating URL with ID {id_url}: {str(e)}")
