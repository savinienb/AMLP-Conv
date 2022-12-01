
import os
import shutil
import sys


def create_directories_for_file_name(file_name):
    """
    Create missing directories for the given file name.
    :param file_name: The file name.
    """
    dir_name = os.path.dirname(file_name)
    create_directories(dir_name)


def create_directories(dir_name):
    """
    Create missing directories.
    :param dir_name: The directory name.
    """
    if dir_name == '':
        return
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def copy_files_to_folder(files, dir_name):
    """
    Copy files to a directory.
    :param files: List of files to copy.
    :param dir_name: The directory name.
    """
    create_directories(dir_name)
    for file_to_copy in files:
        shutil.copy(file_to_copy, dir_name)


class Tee(object):
    """
    Object that can write to multiple files at a time.
    """
    def __init__(self, *files):
        """
        Initializer.
        :param files: List of files to write to.
        """
        self.files = files

    def write(self, obj):
        """
        Write object to files.
        :param obj: The object to write.
        """
        for f in self.files:
            f.write(obj)

    def flush(self):
        """
        Flush file objects.
        """
        for f in self.files:
            f.flush()

    def __getattr__(self, attr):
        """
        Get attribute. This method 'forwards' any attribute access to the stdout located at self.files[0].
        This method can be understood as a workaround to make wandb work which accesses 'encoding' during initialization.
        :param attr: Any attribute, i.e., memeber variable, of the Tee object.
        """
        return getattr(self.files[0], attr)
