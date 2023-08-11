import requests
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

def download_zip(url, destination_path=None):
    """Download a ZIP file from a URL and extract it to a destination directory.
    
    Parameters:
    -----------
    url : str
        The URL of the ZIP file to download.
    destination_path : str, Path
        The path to which the ZIP file will be extracted. If None, the folder name will be taken
        from the last part of the URL path and downloaded to the current working directory.

    Returns:
    --------
    bool
        True if the ZIP file was downloaded and extracted successfully, False otherwise.
    """
    if destination_path is None:
        # Name the ZIP file to be downloaded after the last part of the URL path
        url_parts = urlsplit(url)
        destination_path = Path(url_parts.path).name
    
    # Ensure that the destination path is a Path object ending with ".zip"
    zip_filepath = Path(destination_path)
    if zip_filepath.suffix != ".zip":
        zip_filepath = Path(zip_filepath.parent, zip_filepath.name + ".zip")

    # Create the destination directory if it does not exist
    destination_directory = Path(destination_path).parent
    Path(destination_directory).mkdir(parents=True, exist_ok=True)

    # Download the ZIP file
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