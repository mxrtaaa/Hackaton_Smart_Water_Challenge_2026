from pathlib import Path

def get_project_root() -> Path:
  current = Path(__file__).resolve()
  for parent in [current] + list(current.parents):
    # Znacznik katalogu głównego projektu:
    if (parent / ".git").exists() or (parent / "data").exists():
      return parent
  raise FileNotFoundError("Nie znaleziono katalogu głównego projektu!")