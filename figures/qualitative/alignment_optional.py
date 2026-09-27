"""Optional alignment-only QA adapter. Plotting operations are unchanged."""
import warnings
def require_matplotlib_panel_alignment(*args, **kwargs):
    warnings.warn("External alignment-only QA utility is not bundled; exported figure must be visually checked.", RuntimeWarning)
