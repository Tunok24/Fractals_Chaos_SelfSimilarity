from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

class TqdmWriter(FFMpegWriter):
    def __init__(self, *args, **kwargs):
        self.pbar = None
        super().__init__(*args, **kwargs)

    def setup(self, fig, outfile, dpi, *args, **kwargs):
        # Guess number of frames from animation
        anim = getattr(fig, '_animation', None)
        if anim is not None:
            try:
                nframes = len(anim._framedata)
                self.pbar = tqdm(total=nframes, desc="Saving animation")
            except Exception:
                pass  # gracefully fall back if framedata not available
        return super().setup(fig, outfile, dpi, *args, **kwargs)

    def grab_frame(self, **kwargs):
        if self.pbar is not None:
            self.pbar.update(1)
        return super().grab_frame(**kwargs)

    def finish(self):
        if self.pbar is not None:
            self.pbar.close()
        return super().finish()
