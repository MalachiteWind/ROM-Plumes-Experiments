from plumex.post.edge_figs import trial_lookup_key
from plumex.config import data_lookup
from plumex.video_digest import _load_video
import mitosis
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from plumex.regression_pipeline import _add_regression_to_plot
from plumex.video_digest import _plot_frame
from PIL import Image

trials_folder = Path(__file__).absolute().parents[2] / "trials"

image_folder = Path(__file__).absolute().parents[2] / "paper_images"

img_range = (500,1650)
video, orig_center_fc = _load_video(data_lookup["filename"]["low-864"])

center_points_step, dynamics_step = mitosis.load_trial_data(hexstr="343a81", trials_folder=trials_folder/"dynamics")

X_train_sim = dynamics_step["X_train_sim"]

transform = dynamics_step["scalar_transform"]

X_train_sim_unscaled = transform.inverse_transform(X_train_sim)

trim_video = video[img_range[0]:img_range[1]]

idx= 100

frame = trim_video[idx]
y_lim, x_lim = frame.shape

a,b,c = X_train_sim[idx]


x_lin = np.linspace(1,x_lim,101)
y = -np.sqrt((-x_lin - c) / a + b**2 / (4 * a**2)) - b / (2 * a)

start_gif_frame = 0
end_gif_frame = 200
gif_frames = []
for idx in range(start_gif_frame,end_gif_frame):
    fig, ax = plt.subplots(dpi=300)

    _plot_frame(ax=ax,image=trim_video[idx])

    _add_regression_to_plot(
        ax=ax,
        coeffs=X_train_sim_unscaled[idx],
        method='poly_inv',
        frame_points=np.vstack((x_lin,x_lin,x_lin)).T,
        origin_fc=orig_center_fc,
        y_max=1080,
        color='blue'
    )

    # for flipping y_data
    ymax=1080
    for line in ax.get_lines():
        line.set_ydata(ymax - line.get_ydata())

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(),dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gif_frames.append(Image.fromarray(image))

    plt.close(fig)

gif_frames[0].save(
    image_folder / "general/864sindy.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=200,  # Duration of each frame in milliseconds
    loop=0,  # Loop infinitely
)


