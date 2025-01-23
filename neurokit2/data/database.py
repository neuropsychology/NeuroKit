import pathlib
import urllib
import zipfile


def download_from_url(url, destination_path=None):
    """**Download Files from URLs**

    Download a file from the given URL and save it to the destination path.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    destination_path : str, Path
        The path to which the file will be downloaded. If None, the file name will be taken
        from the last part of the URL path and downloaded to the current working directory.

    Returns
    -------
    bool
        True if the file was downloaded successfully, False otherwise.
    """
    destination_path = _download_path_sanitize(url, destination_path)

    # Download the file
    response = urllib.request.urlopen(url)

    if response.status == 200:
        with destination_path.open("wb") as file:
            file.write(response.read())
        return True
    else:
        return False


def download_zip(url, destination_path=None, unzip=True):
    """**Download ZIP files**

    Download a ZIP file from a URL and extract it to a destination directory.

    Parameters
    ----------
    url : str
        The URL of the ZIP file to download.
    destination_path : str, Path
        The path to which the ZIP file will be extracted. If None, the folder name will be taken
        from the last part of the URL path and downloaded to the current working directory.
    unzip : bool
        Whether to unzip the file or not. Defaults to True.

    Returns
    -------
    bool
        True if the ZIP file was downloaded successfully, False otherwise.
    """
    destination_path = _download_path_sanitize(url, destination_path)
    destination_directory = pathlib.Path(destination_path).parent

    # Ensure that the destination path is a Path object ending with ".zip"
    zip_filepath = pathlib.Path(destination_path)
    if zip_filepath.suffix != ".zip":
        zip_filepath = pathlib.Path(zip_filepath.parent, zip_filepath.name + ".zip")

    # Download the ZIP file
    download_successful = download_from_url(url, zip_filepath)
    if download_successful:
        if unzip:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                extracted_folder_name = pathlib.Path(zip_ref.namelist()[0]).parts[0]

                # Extract the contents
                zip_ref.extractall(destination_directory)

                # Rename the extracted folder to the desired name
                extracted_folder_path = destination_directory / extracted_folder_name
                new_folder_path = (
                    destination_directory / pathlib.Path(destination_path).name
                )
                extracted_folder_path.rename(new_folder_path)

            # Clean up by removing the downloaded ZIP file
            zip_filepath.unlink()

        return True
    else:
        return False


def _download_path_sanitize(url, destination_path=None):
    """Sanitize the destination path of a file to be downloaded from a URL.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    destination_path : str, Path
        The path to which the file will be downloaded. If None, the file name will be taken
        from the last part of the URL path and downloaded to the current working directory.

    Returns
    -------
    Path
        The sanitized destination path.
    """
    if destination_path is None:
        # Name the file to be downloaded after the last part of the URL path
        url_parts = urllib.parse.urlsplit(url)
        destination_path = pathlib.Path(url_parts.path).name

    # Ensure that the destination path is a Path object
    destination_path = pathlib.Path(destination_path)

    # Create the destination directory if it does not exist
    destination_directory = destination_path.parent
    pathlib.Path(destination_directory).mkdir(parents=True, exist_ok=True)

    return destination_path
