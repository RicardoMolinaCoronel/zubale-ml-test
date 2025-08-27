# tests/conftest.py
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))  # .../tests
ROOT = os.path.dirname(ROOT)                       # repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
