# Additional figures and plots for AGU24
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mitosis
from plumex.post.points import _PlotData
from rom_plumes.models import get_contour
from rom_plumes.models import flatten_edge_points
from rom_plumes.concentric_circle import concentric_circle
from matplotlib.patches import Circle
from plumex.plotting import CMAP
from plumex.regress_edge import create_sin_func



from plumex.post.edge_figs import _unpack_data
from plumex.post.edge_figs import trial_lookup_key
from plumex.post.post_utils import _construct_rxy_from_center_fit
from plumex.post.post_utils import apply_theta_shift
from plumex.post.post_utils import RegressionData
from plumex.post.post_utils import _get_default_args
from plumex.regression_pipeline import _construct_rxy_f
from plumex.post.points import _mini_video_digest
from plumex.post.points import _single_img_range
from plumex.video_digest import _plot_frame
from plumex.video_digest import _plot_contours
from plumex.video_digest import _plot_learn_path

image_folder = Path(__file__).absolute().parents[2] / "paper_images"
trials_folder = Path(__file__).absolute().parents[2] / "trials"

def run():
    dpi=300
    # med 914
    print("Creating Rom gif.")
    video_data = _unpack_data(*trial_lookup_key["med 914"]["default"])
    _create_raw_rom_gif(
        video_data=video_data, 
        frames=[i for i in range(250,300)],
        rom=True,
        save_path=image_folder / "agu/agu_rom.gif",
        dpi=dpi
    )
    
    print("Creating raw gif.")
    _create_raw_rom_gif(
        video_data=video_data, 
        frames=[i for i in range(250,300)],
        rom=False,
        save_path=image_folder / "agu/agu_raw.gif"
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
    _create_method_step_gifs(plot_data,save_path=image_folder/"agu",dpi=dpi)

    _, _, edge_regress_hash = trial_lookup_key["med 914"]["default"]
    center_hash = "da0e48" # 
    frame_id = 1000 # 750, 800, 1000,1100
    _create_edge_regress_frame(
        edge_regress_hash, 
        center_hash, 
        frame_id,
        dpi=dpi,
        save_path=image_folder/"agu"
    )
    



def _create_raw_rom_gif(
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
            ax.plot(top_points[:, 0], top_points[:, 1], c="b")
            ax.plot(bot_points[:, 0], bot_points[:, 1], c="g")
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



def _create_method_step_gifs(plot_data: _PlotData, save_path:str,dpi:int=300):
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


def _create_edge_regress_frame(
        edge_regress_hash: str, 
        center_hash: str, 
        frame_id: int,
        save_path: str,
        dpi: int = 300
):
    edge_data = mitosis.load_trial_data(
        hexstr=edge_regress_hash,
        trials_folder=trials_folder / "edge-regress"
    )

    # Get frame with points and concentric circles
    center_data_kws = mitosis._load_trial_params(
        hexstr=center_hash,
        trials_folder=trials_folder / "center"
    )
    start_frame = edge_data[0]["data"]["center"][0][0]
    center_data_kws = _single_img_range(center_data_kws, frame=start_frame + frame_id)
    plot_data = _mini_video_digest(**center_data_kws)

    # Create first plot (clean image with concentric circles)
    fig1, ax1 = plt.subplots(dpi=dpi, figsize=(12, 6))
    clean_img = plot_data.clean_im
    center_pts = plot_data.center
    bottom_pts = plot_data.bottom
    top_pts = plot_data.top

    y_lim, x_lim = clean_img.shape

    _plot_learn_path(
        ax=ax1,
        image=clean_img,
        frame_center=center_pts[0][1],
        frame_top=top_pts[0][1],
        frame_bottom=bottom_pts[0][1],
        marker_size=10
    )

    orig_center = plot_data.orig_center
    concentric_circle_kws = center_data_kws.get(
        "center_kws", _get_default_args(concentric_circle)
    )

    radii = concentric_circle_kws["radii"]
    num_of_circles = concentric_circle_kws["num_of_circs"]

    for radius in range(radii, num_of_circles * radii, radii):
        ax1.add_patch(
            Circle(orig_center, radius, color=CMAP[4], fill=False, linewidth=2)
        )

    ax1.set_xlim([0, x_lim])
    ax1.set_ylim([y_lim, 0])

    # Save first plot
    fig1.savefig(save_path/"pnts_concentric_circ.png", bbox_inches="tight")
    plt.close(fig1)

    # Create second plot (edge regression)
    fig2, ax2 = plt.subplots(dpi=dpi, figsize=(8, 6))
    rad_dist = flatten_edge_points(
        mean_points=center_pts[0][1],
        vari_points=bottom_pts[0][1],
    )
    r_max = np.max(rad_dist[:, 0])
    r_lin = np.linspace(0, r_max, 101)
    time = center_pts[0][0]
    t_lin = np.ones(len(r_lin)) * time

    bot_coef_sin = np.nanmean(
        edge_data[1]["accs"]["bot"]["sinusoid"]["coeffs"], axis=0
    )
    bot_sin_func = create_sin_func(bot_coef_sin)
    bot_sin_vals = bot_sin_func(t_lin, r_lin)
    ax2.plot(r_lin, bot_sin_vals, c='k', linestyle="--", zorder=3)
    ax2.scatter(rad_dist[:, 0], rad_dist[:, 1], c="b", zorder=2)
    ax2.scatter(rad_dist[:, 0], [0] * len(rad_dist[:, 1]), c='r', zorder=2)
    ax2.set_title(r"$d(r,t)=A \sin( \omega r - \gamma t + B) + C + r D$", fontsize=18)
    ax2.hlines(
        y=0, xmin=np.min(rad_dist[:, 0]), xmax=np.max(rad_dist[:, 0]) - 10.0, colors="k",
        zorder=1
    )
    ax2.vlines(
        x=rad_dist[:, 0], ymin=0, ymax=rad_dist[:, 1], colors=CMAP[4],
        linewidth=2, zorder=1
    )

    ax2.set_xlabel("r", fontsize=18)
    ax2.set_ylabel("d", fontsize=18)
    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Save second plot
    fig2.savefig(save_path / "edge_regress.png", bbox_inches="tight")
    plt.close(fig2)



if __name__ == "__main__":
    run()
