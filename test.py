# run inside Python (venv)
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Ensure matplotlib knows where ffmpeg is: prefer system ffmpeg on PATH, else fall back to imageio-ffmpeg
ffmpeg_path = shutil.which('ffmpeg')
if ffmpeg_path is None:
    try:
        import imageio_ffmpeg as iio
        ffmpeg_path = iio.get_ffmpeg_exe()
    except Exception:
        ffmpeg_path = None

if ffmpeg_path:
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    print(f"Using ffmpeg at: {ffmpeg_path}")
else:
    print("Warning: ffmpeg not found. Install system ffmpeg or imageio-ffmpeg to enable MP4 writing.")

frames = [np.random.randint(0,255,(64,64,3),dtype='uint8') for _ in range(20)]
fig = plt.figure()
writer = FFMpegWriter(fps=10)
with writer.saving(fig, "test_out.mp4", dpi=100):
    for f in frames:
        plt.imshow(f);
        plt.axis('off');
        writer.grab_frame()
plt.close(fig)
print("test_out.mp4 written")