from typing import Optional
from app.core.dependencies import db_tfm as db


async def get_domains(domain: Optional[str] = None, id_domain: Optional[int] = None):
    """
    Fetch all domains or filter by a specific domain.
    If domain is provided, it returns only that domain; otherwise, it returns all domains.
    """
    try:
        query = "SELECT * FROM domains"
        if domain:
            return await db.fetch_one(query + " WHERE domain_url like ?", (domain,))
        elif id_domain is not None:
            return await db.fetch_one(query + " WHERE id_domain = ?", (id_domain,))
        else:
            return await db.fetch_all(query, params=None)
    except Exception as e:
        raise RuntimeError(f"Error fetching domains: {str(e)}")


async def save_domain(domain_url: str, absolute_url: str):
    """
    Save a new domain to the database.
    If the domain already exists, it returns the existing domain.
    """
    try:
        existing_domain = await get_domains(domain=domain_url)
        if existing_domain:
            return existing_domain

        new_domain = {"domain_url": domain_url, "absolute_url": absolute_url}
        return await db.insert("domains", new_domain, return_id=True)

    except Exception as e:
        raise RuntimeError(f"Error saving domain {domain_url}: {str(e)}")


async def delete_domain(id_domain: int):
    """
    Delete a domain by its ID.
    """
    try:
        return await db.delete("domains", where_conditions={"id_domain": id_domain})
    except Exception as e:
        raise RuntimeError(f"Error deleting domain with ID {id_domain}: {str(e)}")


async def update_domain(
    id_domain: int, domain_url: Optional[str] = None, absolute_url: Optional[str] = None
):
    """
    Update an existing domain's URL or absolute URL.
    If the domain does not exist, it raises an error.
    """
    try:
        updates = {}
        if domain_url:
            updates["domain_url"] = domain_url
        if absolute_url:
            updates["absolute_url"] = absolute_url

        if not updates:
            raise ValueError("No fields to update provided.")

        return await db.update(
            "domains", set_fields=updates, where_conditions={"id_domain": id_domain}
        )

    except Exception as e:
        raise RuntimeError(f"Error updating domain with ID {id_domain}: {str(e)}")
