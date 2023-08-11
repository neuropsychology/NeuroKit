import requests
import zipfile
import pathlib
import urllib.parse

def download_zip(url, destination_path=None, unzip=True):
    """Download a ZIP file from a URL and extract it to a destination directory.
    
    Parameters:
    -----------
    url : str
        The URL of the ZIP file to download.
    destination_path : str, Path
        The path to which the ZIP file will be extracted. If None, the folder name will be taken
        from the last part of the URL path and downloaded to the current working directory.
    unzip : bool
        Whether to unzip the file or not. Defaults to True.

    Returns:
    --------
    bool
        True if the ZIP file was downloaded successfully, False otherwise.
    """
    if destination_path is None:
        # Name the ZIP file to be downloaded after the last part of the URL path
        url_parts = urllib.parse.urlsplit(url)
        destination_path = pathlib.Path(url_parts.path).name

    # Ensure that the destination path is a Path object ending with ".zip"
    zip_filepath = pathlib.Path(destination_path)
    if zip_filepath.suffix != ".zip":
        zip_filepath = pathlib.Path(zip_filepath.parent, zip_filepath.name + ".zip")

    # Create the destination directory if it does not exist
    destination_directory = pathlib.Path(destination_path).parent
    pathlib.Path(destination_directory).mkdir(parents=True, exist_ok=True)

    # Download the ZIP file
    response = requests.get(url)

    if response.status_code == 200:
        with zip_filepath.open("wb") as zip_file:
            zip_file.write(response.content)

        if unzip:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                extracted_folder_name = pathlib.Path(zip_ref.namelist()[0]).parts[0]
                    
                # Extract the contents
                zip_ref.extractall(destination_directory)
                
                # Rename the extracted folder to the desired name
                extracted_folder_path = destination_directory / extracted_folder_name
                new_folder_path = destination_directory / pathlib.Path(destination_path).name
                extracted_folder_path.rename(new_folder_path)

            # Clean up by removing the downloaded ZIP file
            zip_filepath.unlink()

        return True
    else:
        return False