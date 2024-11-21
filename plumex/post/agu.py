# Additional figures and plots for AGU24
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mitosis
from plumex.post.points import _PlotData
from rom_plumes.models import get_contour



from plumex.post.edge_figs import _unpack_data
from plumex.post.edge_figs import trial_lookup_key
from plumex.post.post_utils import _construct_rxy_from_center_fit
from plumex.post.post_utils import apply_theta_shift
from plumex.post.post_utils import RegressionData
from plumex.regression_pipeline import _construct_rxy_f
from plumex.post.points import _mini_video_digest
from plumex.post.points import _single_img_range
from plumex.video_digest import _plot_frame
from plumex.video_digest import _plot_contours
from plumex.video_digest import _plot_learn_path

image_folder = Path(__file__).absolute().parents[2] / "paper_images"
trials_folder = Path(__file__).absolute().parents[2] / "trials"

def run():
    # med 914
    print("Creating Rom gif.")
    video_data = _unpack_data(*trial_lookup_key["med 914"]["default"])
    _create_frames(
        video_data=video_data, 
        frames=[i for i in range(250,300)],
        rom=True,
        save_path=image_folder / "agu_rom.gif"
    )
    
    print("Creating raw gif.")
    _create_frames(
        video_data=video_data, 
        frames=[i for i in range(250,300)],
        rom=False,
        save_path=image_folder / "agu_raw.gif"
    )

    
    # Refer to config in center_regress_hash config 
    # get center hash
    center_hash = "da0e48" # maybe automate this

    center_data_kws = mitosis._load_trial_params(
        hexstr=center_hash,
        trials_folder=trials_folder / "center"
    )

    start_frame = video_data["center_plume_points"][0][0]
    center_data_kws = _single_img_range(
        center_data_kws,frame=[i + start_frame for i in range(250,300)]
    )

    plot_data = _mini_video_digest(**center_data_kws)

    print("creating steps gif.")
    method_frames(plot_data,save_path=image_folder)




def _create_frames(
    video_data: RegressionData,
    frames: List[int],
    rom: bool,
    save_path: str,
    dpi:int=300
) -> None:
    gif_frames = []

    for idx in frames:
        # Create a new figure
        fig, ax = plt.subplots(dpi=dpi)

        frame_t = video_data["video"][idx]
        y_lim, x_lim = frame_t.shape

        # Display the video frame
        ax.imshow(frame_t, cmap="gray")
        if rom:
            # Plot center points and fit
            raw_center_points = video_data["center_plume_points"][idx][1]
            orig_center_fc = video_data["orig_center_fc"]
            center_fit_method = video_data["center_func_method"]
            center_fit_func = _construct_rxy_f(
                coef=video_data["center_coef"][idx], regression_method=center_fit_method
            )
            raw_center_points[:, 1:] -= orig_center_fc
            if center_fit_method == "poly_para":
                r_min = np.min(raw_center_points[:, 0]) * 1.5
                r_max = np.max(raw_center_points[:, 0]) * 1.5
                r_vals = np.linspace(r_min, r_max, 101)
                fit_centerpoints_dc = _construct_rxy_from_center_fit(
                    indep_data=r_vals,
                    center_fit_func=center_fit_func,
                    regression_method=center_fit_method,
                )
            else:
                x_min = np.min(raw_center_points[:, 1]) * 2
                x_max = np.max(raw_center_points[:, 1]) * 1.5
                x_vals = np.linspace(x_min, x_max, 101)
                fit_centerpoints_dc = _construct_rxy_from_center_fit(
                    indep_data=x_vals,
                    center_fit_func=center_fit_func,
                    regression_method=center_fit_method,
                )
            raw_center_points[:, 1:] += orig_center_fc
            fit_centerpoints_dc[:, 1:] += orig_center_fc

            # Plot the fitted center points
            ax.plot(fit_centerpoints_dc[:, 1], fit_centerpoints_dc[:, 2], c="r")

            # Compute and plot top and bottom points
            anchor_points = fit_centerpoints_dc
            top_points = []
            bot_points = []
            for rad, x_fc, y_fc in anchor_points:
                top_points.append(
                    apply_theta_shift(
                        t=idx,
                        r=rad,
                        x=x_fc - orig_center_fc[0],
                        y=y_fc - orig_center_fc[1],
                        flat_regress_func=video_data["top_edge_func"],
                        positive=True,
                    )
                )
                bot_points.append(
                    apply_theta_shift(
                        t=idx,
                        r=rad,
                        x=x_fc - orig_center_fc[0],
                        y=y_fc - orig_center_fc[1],
                        flat_regress_func=video_data["bot_edge_func"],
                        positive=False,
                    )
                )

            top_points = np.array(top_points) + orig_center_fc
            bot_points = np.array(bot_points) + orig_center_fc

            # Plot the top and bottom edges
            ax.plot(top_points[:, 0], top_points[:, 1], c="g")
            ax.plot(bot_points[:, 0], bot_points[:, 1], c="b")
        ax.set_xlim([0, x_lim])
        ax.set_ylim([y_lim, 0])
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # Save the figure as an image in memory
        fig.canvas.draw()  # Render the figure
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(Image.fromarray(image))

        # Close the figure to avoid display overlap
        plt.close(fig)

    # Save all collected frames as a GIF
    gif_frames[0].save(
        save_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,  # Duration of each frame in milliseconds
        loop=0,        # Loop infinitely
    )



def method_frames(plot_data: _PlotData, save_path:str,dpi:int=300):
    # SAVE PATHs do it 

    # exptup = plot_data

    # Access raw frames
    raw_frames = plot_data.raw_im
    gif_frames = []
    for frame in raw_frames:
        fig, ax = plt.subplots(dpi=dpi)
        _plot_frame(ax,frame)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(Image.fromarray(image))
        plt.close(fig)

    gif_frames[0].save(
        save_path / "raw_step.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,  # Duration of each frame in milliseconds
        loop=0,        # Loop infinitely
    )


    # Access clean frames
    clean_frames = plot_data.clean_im
    gif_frames = []
    for frame in clean_frames:
        fig, ax = plt.subplots(dpi=dpi)

        _plot_frame(ax,frame)
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(Image.fromarray(image))
        plt.close(fig)

    gif_frames[0].save(
        save_path / "clean_step.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,  # Duration of each frame in milliseconds
        loop=0,        # Loop infinitely
    )


    # Access Contour and cocentric circle
    orig_center = plot_data.orig_center
    gif_frames = []
    for frame in clean_frames:
        fig, ax = plt.subplots(dpi=dpi)

        contours = get_contour(frame,**plot_data.contour_kws)

        _plot_contours(
            ax,
            frame,
            orig_center,
            contours
        )

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(Image.fromarray(image))
        plt.close(fig)
    
    gif_frames[0].save(
        save_path / "contour_step.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,  # Duration of each frame in milliseconds
        loop=0,        # Loop infinitely
    )


    # Access blur with center, top, bot
    center = plot_data.center
    bottom = plot_data.bottom
    top = plot_data.top
    gif_frames = []

    for idx, frame in enumerate(clean_frames):
        fig, ax = plt.subplots(dpi=dpi)

        _plot_learn_path(
            ax=ax,
            image=frame,
            frame_center=center[idx][1],
            frame_top=top[idx][1],
            frame_bottom=bottom[idx][1],
            marker_size=10
        )

        fig.canvas.draw()  # Render the figure
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(Image.fromarray(image))
        plt.close(fig)
    
    gif_frames[0].save(
        save_path / "rom_step.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,  # Duration of each frame in milliseconds
        loop=0,        # Loop infinitely
    )


if __name__ == "__main__":
    run()
