
import squidpy as sq
adata = sq.datasets.visium_hne_adata()
img = sq.datasets.visium_hne_image()
viewer = img.interactive(adata)

viewer.screenshot(canvas_only=False)