import os
import traceback


def get_main_file_dir() -> str:
  """Get the main file directory from the traceback.

  Returns:
      str: The main file directory.
  """
  main_file_path = traceback.extract_stack()[0].filename
  main_file_dir = os.path.dirname(main_file_path)
  return main_file_dir
