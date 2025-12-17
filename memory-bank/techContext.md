Python 3.9+, PyQt6/qtpy, pyqtgraph, numpy, scipy, matplotlib, PyOpenGL.
Notes:
- `PyOpenGL_accelerate` was removed from `requirements.txt` due to driver/context segfaults on some Linux setups; Matplotlib is used for stable 3D rendering instead of relying on accelerated OpenGL.
- The GUI is built with PyQt components and visualization is handled primarily by Matplotlib and pyqtgraph (2D). OpenGL usage has been minimized.