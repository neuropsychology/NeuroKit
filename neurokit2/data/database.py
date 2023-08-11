import requests
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

def download_zip(url, destination_directory, zip_filename=None):
    """Download a ZIP file from a URL and extract it to a destination directory.
    
    Parameters:
    -----------
    url : str
        The URL of the ZIP file to download.
    destination_directory : str, Path
        The directory to which the ZIP file will be extracted.
    zip_filename : str, optional
        The name of the ZIP file to download. If None, the name will be taken from the last part of the URL path.

    Returns:
    --------
    bool
        True if the ZIP file was downloaded and extracted successfully, False otherwise.
    """
    # Create the destination directory if it does not exist
    Path(destination_directory).mkdir(parents=True, exist_ok=True)

    if zip_filename is None:
        # Name the ZIP file to be downloaded after the last part of the URL path
        url_parts = urlsplit(url)
        zip_filename = Path(url_parts.path).name

    # Download the ZIP file
    zip_filepath = Path(destination_directory) / zip_filename
    response = requests.get(url)

    if response.status_code == 200:
        with zip_filepath.open("wb") as zip_file:
            zip_file.write(response.content)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(destination_directory)

        # Clean up by removing the downloaded ZIP file
        zip_filepath.unlink()

        return True
    else:
        return False