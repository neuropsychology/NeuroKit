import requests
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

def download_zip(url, destination_directory):
    """Download a ZIP file from a URL and extract it to a destination directory.
    
    Parameters:
    -----------
    url : str
        The URL of the ZIP file to download.
    destination_directory : str, Path
        The directory to which the ZIP file will be extracted.

    Returns:
    --------
    bool
        True if the ZIP file was downloaded and extracted successfully, False otherwise.
    """
    # Create the destination directory if it does not exist
    Path(destination_directory).mkdir(parents=True, exist_ok=True)

    # Name the ZIP file to be downloaded after the last part of the URL path
    url_parts = urlsplit(url)
    filename = Path(url_parts.path).name

    # Download the ZIP file
    zip_filename = destination_directory / filename
    response = requests.get(url)

    if response.status_code == 200:
        with zip_filename.open("wb") as zip_file:
            zip_file.write(response.content)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(destination_directory)

        # Clean up by removing the downloaded ZIP file
        zip_filename.unlink()

        return True
    else:
        return False