
from Lens2Image_toy_model import LensReconstruction

if __name__ == '__main__':

    Gan = LensReconstruction()
    real_1 = Gan.lens2source_train(
                       epochs=500, batch_size=64,
                       train_im_file="S_300_10000.npy",
                       train_lb_file="M_300_10000.npy",
                       progress=True, progress_interval=10, progress_save=True,
                       progress_file="TOYresult/progress.log",
                       plot_image=True,
                       save_iter=50, plot_save_iter=10, save_plots_path="TOYresult", save_plots_type="png",
                       savegfile="TOYresult/Generator",
                       savedfile="TOYresult/Descriminator",
                       )
