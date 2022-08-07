"""

This is the main training file for SIE model

CHENGYI

"""

from Lens2Image import LensReconstruction

if __name__ == '__main__':

    # data[0] source ; data[1] lens
    datapass = ["gray", "lensed"]

    Gan = LensReconstruction(datapass, datapass,
                      img_size=(64, 64, 1),
                      lbl_size=(64, 64, 1))
    real_1 = Gan.lens2source_train(
                       epochs=500, batch_size=64,
                       train_lb_path="lensed_shrink_64",
                       train_im_path="img_shrink_64",
                       progress=True, progress_interval=10, progress_save=True,
                       progress_file="SIEresult_64/progress.log",
                       plot_image=True, plot_save_iter=10,
                       save_plots_path="SIEresult_64", save_plots_type="png",
                       save_iter=50,
                       savegfile="SIEresult_64/Generator",
                       savedfile="SIEresult_64/Descriminator",
                       )
