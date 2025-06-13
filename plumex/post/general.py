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

# Folders
trials_folder = Path(__file__).absolute().parents[2] / "trials"
image_folder = Path(__file__).absolute().parents[2] / "paper_images"

# (file_name, dynamics_hash, video_range)
sim_dict = {
    "864": ("low-864", "343a81", (500,1650)),
    "862": ("low-862", "8495e5",(1000,1450))
}

img_range = (500,1650)
video, orig_center_fc = _load_video(data_lookup["filename"]["low-864"])

def run():
    # # 862
    # trim_video, X_train_sim, orig_center_fc = _load_data("862")

    # create_sindy_gif(
    #     vid=trim_video,
    #     sindy_coef=X_train_sim, 
    #     gif_range=[0,80],
    #     orig_center_fc=orig_center_fc,
    #     save_path=image_folder / "general/862_sindy.gif"
    # )

    trim_video, X_train_sim, orig_center_fc = _load_data("864")

    create_sindy_gif(
        vid=trim_video,
        sindy_coef=X_train_sim,
        gif_range=[20,120],
        orig_center_fc=orig_center_fc,
        save_path=image_folder / "general/864_sindy.gif"
    )
    


def _load_data(sim_dict_key):
    vid_kw, dynamics_hash, trim_range = sim_dict[sim_dict_key]
    video, orig_center_fc = _load_video(data_lookup["filename"][vid_kw])

    trim_video = video[trim_range[0]: trim_range[1]]


    _, dynamics_step = mitosis.load_trial_data(
        hexstr=dynamics_hash,
        trials_folder=trials_folder/"dynamics"
    )

    X_train_sim = dynamics_step["X_train_sim"]

    transform = dynamics_step["scalar_transform"]

    sindy_model = dynamics_step["model"]
    sindy_model.print()

    X_train_sim_unscaled = transform.inverse_transform(X_train_sim)

    return trim_video, X_train_sim_unscaled, orig_center_fc



def create_sindy_gif(vid, sindy_coef,gif_range,orig_center_fc, save_path):
    _, x_max = vid[0].shape

    x_lin = np.linspace(1,x_max, 101)
    gif_frames=[]
    for idx in range(gif_range[0], gif_range[1]):
        fig, ax = plt.subplots(dpi=200)

        _plot_frame(ax=ax, image=vid[idx])

        _add_regression_to_plot(
            ax=ax,
            coeffs=sindy_coef[idx],
            method='poly_inv',
            frame_points=np.vstack((x_lin,x_lin,x_lin)).T,
            origin_fc=orig_center_fc,
            y_max=1080,
            color='red'
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
        save_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=100,  # Duration of each frame in milliseconds
        loop=0,  # Loop infinitely
    )    


if __name__ == "__main__":
    run()

# center_points_step, dynamics_step = mitosis.load_trial_data(hexstr="343a81", trials_folder=trials_folder/"dynamics")

# X_train_sim = dynamics_step["X_train_sim"]

# transform = dynamics_step["scalar_transform"]

# X_train_sim_unscaled = transform.inverse_transform(X_train_sim)

# trim_video = video[img_range[0]:img_range[1]]

# idx= 100

# frame = trim_video[idx]
# y_lim, x_lim = frame.shape

# a,b,c = X_train_sim[idx]


# x_lin = np.linspace(1,x_lim,101)

# start_gif_frame = 0
# end_gif_frame = 200
# gif_frames = []
# for idx in range(start_gif_frame,end_gif_frame):
#     fig, ax = plt.subplots(dpi=300)

#     _plot_frame(ax=ax,image=trim_video[idx])

#     _add_regression_to_plot(
#         ax=ax,
#         coeffs=X_train_sim_unscaled[idx],
#         method='poly_inv',
#         frame_points=np.vstack((x_lin,x_lin,x_lin)).T,
#         origin_fc=orig_center_fc,
#         y_max=1080,
#         color='blue'
#     )

#     # for flipping y_data
#     ymax=1080
#     for line in ax.get_lines():
#         line.set_ydata(ymax - line.get_ydata())

#     fig.canvas.draw()
#     image = np.frombuffer(fig.canvas.tostring_rgb(),dtype="uint8")
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     gif_frames.append(Image.fromarray(image))

#     plt.close(fig)

# gif_frames[0].save(
#     image_folder / "general/864sindy.gif",
#     save_all=True,
#     append_images=gif_frames[1:],
#     duration=200,  # Duration of each frame in milliseconds
#     loop=0,  # Loop infinitely
# )


