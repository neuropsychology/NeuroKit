# -*- coding: utf-8 -*-
#import numpy as np
#
#
# https://towardsdatascience.com/modality-tests-and-kernel-density-estimations-3f349bb9e595
#def getKernelDensityEstimation(values, x, bandwidth = 0.2, kernel = 'gaussian'):
#    model = KernelDensity(kernel = kernel, bandwidth=bandwidth)
#    model.fit(values[:, np.newaxis])
#    log_density = model.score_samples(x[:, np.newaxis])
#    return np.exp(log_density)
#for bandwidth in np.linspace(0.2, 3, 3):
#    kde = getKernelDensityEstimation(data, x, bandwidth=bandwidth)
#    plt.plot(x, kde, alpha = 0.8, label = f'bandwidth = {round(bandwidth, 2)}')
#plt.plot(x, true_pdf, label = 'True PDF')
