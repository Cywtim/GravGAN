"""

This is the main training file for SIE model

CHENGYI

"""

from Lens2Image import LensReconstruction

if __name__ == '__main__':

    # data[0] source ; data[1] lens
    datapass = ["/project/Redshift_Difference/GravGan/gray",
                "/project/Redshift_Difference/GravGan/SIElensed"]

    Gan = LensReconstruction(datapass, datapass,
                      img_size=(256, 256, 1),
                      lbl_size=(256, 256, 1))
    real_1 = Gan.lens2source_train(
                       epochs=5000, batch_size=128,
                       train_lb_path="/project/Redshift_Difference/GravGan/SIElensed",
                       progress=True, progress_interval=50, progress_save=True,
                       progress_file="/project/Redshift_Difference/GravGan/SIEresult/progress_cpu.log",
                       plot_image=True, save_plots=True, plot_save_iter=100,
                       save_plots_path="/project/Redshift_Difference/GravGan/SIEresult",
                       save_plots_type="pdf", save_iter=50,
                       savegfile="/project/Redshift_Difference/GravGan/SIEresult/GeneratorCPU",
                       savedfile="/project/Redshift_Difference/GravGanSIEresult/DescriminatorCPU",
                       )
