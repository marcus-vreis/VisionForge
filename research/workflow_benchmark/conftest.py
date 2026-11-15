"""The published benchmark series hold the original recipe scripts, not tests."""

# real-v1/recipe/det_test.py matches pytest's *_test.py pattern and imports
# Ultralytics at module level; nothing under results/ is a test.
collect_ignore = ["results"]
