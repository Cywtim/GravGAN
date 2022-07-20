
from Lens2Image_toy_model import LensReconstruction

if __name__ == '__main__':

    Gan = LensReconstruction()
    real_1 = Gan.lens2source_train(
                       epochs=100, batch_size=16,
                       train_im_file="S_300_10000.npy",
                       train_lb_path="M_300_10000.npy",
                       progress=True, progress_interval=1, progress_save=True,
                       progress_file="TOYresult/progress.log",
                       plot_image=False,
                       save_iter=50,
                       savegfile="SIEresult/Generator",
                       savedfile="SIEresult/Descriminator",
                       )
