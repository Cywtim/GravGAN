
from Lens2Image_toy_model import LensReconstruction

if __name__ == '__main__':

    Gan = LensReconstruction()
    real_1 = Gan.lens2source_train(
                       epochs=3000, batch_size=128,
                       train_im_file="project/Redshift_Difference/GravGan/S_300_10000.npy",
                       train_lb_file="project/Redshift_Difference/GravGan/M_300_10000.npy",
                       progress=True, progress_interval=50, progress_save=True,
                       progress_file="project/Redshift_Difference/GravGan/TOYresult/progress.log",
                       plot_image=True, plot_save_iter=30,
                       save_plots_path="project/Redshift_Difference/GravGan/TOYresult", save_plots_type="eps",
                       save_iter=100,
                       savegfile="project/Redshift_Difference/GravGan/TOYresult/Generator",
                       savedfile="project/Redshift_Difference/GravGan/TOYresult/Descriminator",
                       )
