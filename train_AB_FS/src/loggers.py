from abc import ABC, abstractmethod


class AbstractLogger(ABC):
    @abstractmethod
    def log(self, message: str):
        """
        Log the provided message.

        Parameters:
        - message (str): The message to be logged.
        """
        pass


class FileLogger(AbstractLogger):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def log(self, message: str):
        with open(self.file_path, "a") as file:
            file.write(message + "\n")
