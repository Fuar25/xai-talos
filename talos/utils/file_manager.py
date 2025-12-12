import sys
import os


def get_main_file_dir() -> str:
  """Get the main file directory of the currently running script.

  Returns:
      str: The main file directory.
  """
  # Get the module object for the entry point
  main_module = sys.modules['__main__']

  # Check if it has a file attribute (it might not in a REPL/shell)
  if hasattr(main_module, '__file__'):
    main_file_path = os.path.abspath(main_module.__file__)
    return os.path.dirname(main_file_path)

  print(" ! Cannot determine main file path in interactive mode.")
  return None