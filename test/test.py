
# Assuming CloudStorage is an abstract base class defined elsewhere.
# from cloud_storage import CloudStorage

class CloudStorageImpl(CloudStorage):
    def __init__(self):
        """
        Initializes the in-memory storage for files.
        The storage is a dictionary where keys are file names (str)
        and values are their sizes (int).
        """
        self.files = {}

    def add_file(self, name: str, size: int) -> bool:
        """
        Adds a new file to the storage.

        Args:
            name: The name of the file (e.g., 'dir1/file.txt').
            size: The size of the file in bytes.

        Returns:
            True if the file was successfully added, False if a file
            with the same name already exists.
        """
        # If the file name is already a key in our dictionary, it exists.
        if name in self.files:
            return False
        
        # Add the new file to the dictionary.
        self.files[name] = size
        return True

    def get_file_size(self, name: str) -> int | None:
        """
        Retrieves the size of a file.

        Args:
            name: The name of the file.

        Returns:
            The size of the file as an integer, or None if the file
            does not exist.
        """
        # The .get() method is a safe way to retrieve a value;
        # it returns None if the key is not found.
        return self.files.get(name)

    def delete_file(self, name: str) -> int | None:
        """
        Deletes a file from the storage.

        Args:
            name: The name of the file to delete.

        Returns:
            The size of the deleted file as an integer, or None if
            the file did not exist.
        """
        # The .pop() method removes the key and returns its value.
        # We use a default value of None, so if the key doesn't exist,
        # it returns None instead of raising an error.
        return self.files.pop(name, None)