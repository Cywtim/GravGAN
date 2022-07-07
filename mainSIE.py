"""

This is the main training file for SIE model

CHENGYI

"""

from Lens2Image import LensReconstruction

if __name__ == '__main__':

    # data[0] source ; data[1] lens
    datapass = ["gray", "SIElensed"]

    Gan = LensReconstruction(datapass, datapass,
                      img_size=(256, 256, 1),
                      lbl_size=(256, 256, 1))
    real_1 = Gan.lens2source_train(
                       epochs=5000, batch_size=32,
                       train_lb_path="SIElensed",
                       progress=True, progress_interval=10, progress_save=True,
                       progress_file="SIEresult/progress.log",
                       plot_image=False,
                       save_iter=50,
                       savegfile="SIEresult/Generator",
                       savedfile="SIEresult/Descriminator",
                       )
