# Image Processing Projects

Two implementations developed during COMP0026 (2025–26):

- **Image morphing:** landmark alignment, Delaunay triangulation, piecewise affine warping, interpolation and a meshless warping variant.
- **Poisson editing:** gradient-domain cloning, mixed gradients, texture flattening, local illumination and colour changes, and tiling.

## Run

Use Python 3.10+ and install `requirements.txt`. The OpenCV editor requires a desktop environment and Tk support:

```bash
python poisson/poisson_editor.py
```

Select your own images in the dialogs, draw the masks and follow the console prompts. The notebook is also available in `poisson/`.

For morphing, start Jupyter from the `morphing/` directory and open `image_morphing.ipynb`. Supply `img1.jpg`, `img2.jpg`, and either your own `lm1_coords.npy`/`lm2_coords.npy` landmarks or an appropriately licensed dlib predictor named `shape_predictor_68_face_landmarks.dat`. These files are deliberately not distributed.

The editor's fixed placement offsets may need adjustment for your image dimensions. This export preserves the original algorithms rather than changing their behaviour.

## Publication scope

This is a curated portfolio of coursework implementations from the author's local working files. Course questions, marking rubrics, slides, reports, student identifiers, notebook outputs, input datasets, trained weights and commercial models are not included. Original private archives remain separate.

Supply your own appropriately licensed inputs where required. Existing algorithm limitations are preserved; publication is not a claim of a new benchmark or a complete reproduction of the original assessment. Dependencies retain their own licences. No blanket licence is added to third-party material.
