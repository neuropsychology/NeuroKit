import requests
import zipfile
from pathlib import Path

def download_zip(url, destination_path):
    """Download a ZIP file from a URL and extract it to a destination directory.
    
    Parameters:
    -----------
    url : str
        The URL of the ZIP file to download.
    destination_path : str, Path
        The path to which the ZIP file will be extracted.

    Returns:
    --------
    bool
        True if the ZIP file was downloaded and extracted successfully, False otherwise.
    """
    # Ensure that the destination path is a Path object ending with ".zip"
    zip_filepath = Path(destination_path)
    if zip_filepath.suffix != ".zip":
        zip_filepath = zip_filepath.with_suffix(".zip")

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