import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET


def get_sitemaps_from_domain(domain: str):
    """
    Get the urls of the sitemaps from a domain, the domain may have several sitemaps, in that case,
    it returns all of them
    """
    # Primero lee el archivo robots.txt para saber si se ha especificado ahí el sitemap
    robots_url = domain + "/robots.txt"
    robots = requests.get(robots_url)
    sitemaps = []
    if robots.status_code == 200:
        for line in robots.text.split("\n"):
            if "sitemap" in line.lower():
                sitemap = line.split(":")[1].strip()
                sitemaps.append(sitemap)

    # Si no se ha encontrado ningún sitemap, se intenta con el archivo sitemap_index.xml
    if not sitemaps:
        sitemap_index_url = domain + "/sitemap_index.xml"
        sitemap_index = requests.get(sitemap_index_url)
        if sitemap_index.status_code == 200:
            for line in sitemap_index.text.split("\n"):
                if "sitemap" in line.lower():
                    sitemap = line.split(":")[1].strip()
                    sitemaps.append(sitemap)

    # Si no se ha encontrado ningún sitemap, se intenta con el archivo sitemap.xml
    if not sitemaps:
        sitemap_url = domain + "/sitemap.xml"
        sitemap = requests.get(sitemap_url)
        if sitemap.status_code == 200:
            sitemaps.append(sitemap_url)

    return sitemaps


def fetch_sitemap_content(url):
    """
    Descarga y devuelve el contenido de un sitemap.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error al acceder al sitemap {url}: {e}")
        return None


def parse_sitemap(content):
    """
    Analiza un sitemap y distingue entre URLs y sub-sitemaps.
    """
    urls = []
    sub_sitemaps = []

    try:
        root = ET.fromstring(content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Buscar etiquetas <loc> dentro de <urlset> (URLs finales)
        for url in root.findall(".//ns:url/ns:loc", namespace):
            if "en/" in url.text:
                urls.append(url.text)

        # Buscar etiquetas <loc> dentro de <sitemapindex> (sub-sitemaps)
        for sitemap in root.findall(".//ns:sitemap/ns:loc", namespace):
            sub_sitemaps.append(sitemap.text)

    except ET.ParseError as e:
        print(f"Error al analizar el contenido del sitemap: {e}")

    return urls, sub_sitemaps


def get_all_urls_and_sitemaps(initial_sitemaps):
    """
    Recorre recursivamente los sitemaps para obtener todas las URLs y sub-sitemaps.
    """
    all_urls = set()
    processed_sitemaps = set()
    pending_sitemaps = set(initial_sitemaps)

    while pending_sitemaps:
        current_sitemap = pending_sitemaps.pop()
        if current_sitemap in processed_sitemaps:
            continue

        print(f"Procesando: {current_sitemap}")
        content = fetch_sitemap_content(current_sitemap)
        if not content:
            continue

        urls, sub_sitemaps = parse_sitemap(content)
        all_urls.update(urls)
        pending_sitemaps.update(sub_sitemaps)
        processed_sitemaps.add(current_sitemap)

    return all_urls, processed_sitemaps


def get_english_urls_from_sitemap(sitemap_url: str):
    """
    Get the urls from a sitemap, it returns only the English urls
    """
    # Get the sitemap
    sitemap = requests.get(sitemap_url)
    urls = []
    if sitemap.status_code == 200:
        for line in sitemap.text.split("\n"):
            if "loc" in line:
                url = line.split("<loc>")[1].split("</loc>")[0]
                if "en/" in url:
                    urls.append(url)
    return urls


def get_url_html_content(url: str):
    """
    Get the content of a url
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error al acceder a la URL {url}: {e}")
        return None


def get_url_texts_content(html_text: str, filter_tag: str = None):
    """
    Extracts text content from HTML, removing scripts, styles, and other non-visible elements.
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Cogemos solo el body del HTML
    soup = soup.find("body")

    # Si tiene main, cogemos solo el main
    if soup and soup.find("main"):
        soup = soup.find("main")

    # Si hay que filtrar por algún tag específico, cogemos solo esos
    if soup and filter_tag and soup.find_all(filter_tag):
        soup = soup.find_all(filter_tag)

    # Get the text content
    text_content = [item.get_text(separator=" ", strip=True) for item in soup]
    # Remove multiple spaces and newlines
    text_content = [
        text.replace("\n", " ").replace(r"\s+", " ").strip() for text in text_content
    ]
    text_content = [text for text in text_content if text]  # Filter out empty strings
    # Split the texts by . to get individual sentences
    text_content = [
        line.replace(r"\s+", " ").strip()
        for text in text_content
        for line in text.split(".")
        if line.replace(r"\s+", " ").strip()
    ]

    return text_content
