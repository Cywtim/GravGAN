"""

This is the main training file for SIE model

CHENGYI

"""

from Lens2ParaImage import LensReconstruction

if __name__ == '__main__':

    # data[0] source ; data[1] lens
    datapass = ["gray", "SIElensed", "SIEpara"]

    Gan = LensReconstruction(datapass, datapass,
                      img_size=(256, 256, 1),
                      lbl_size=(256, 256, 1))
    real_1 = Gan.lens2source_train(
                       epochs=100, batch_size=8,
                       train_lb_path="SIElensed",
                       train_para_path="SIEpara",
                       progress=True, progress_interval=1, progress_save=True,
                       progress_file="SIEresult/progress.log",
                       plot_image=False,
                       save_iter=50,
                       savegfile="SIEresult/Generator",
                       savedfile="SIEresult/Descriminator",
                       )
