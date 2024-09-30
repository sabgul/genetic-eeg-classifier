VALID_FEATURES = [
    # 'app_entropy', #computationally expensive
    'higuchi_fd',
    'hjorth_complexity',
    'hjorth_complexity_spect',
    'hjorth_mobility',
    'hjorth_mobility_spect',
    # 'hurst_exp',  # computationally expensive
    'katz_fd',
    'kurtosis',
    'line_length',
    'mean',
    'pow_freq_bands',
    'ptp_amp',
    'quantile',
    'rms',
    # 'samp_entropy',  # computationally expensive
    'skewness',
    'spect_edge_freq',
    'spect_entropy',
    'spect_slope',
    'std',
    # 'svd_entropy', # computationally expensive
    # 'svd_fisher_info',# computationally expensive
    'teager_kaiser_energy',
    'variance',
    'wavelet_coef_energy',
    'zero_crossings',
    # --- bivariate features
    # 'max_cross_corr', # computationally expensive
    # 'nonlin_interdep', # computationally expensive
    'phase_lock_val',
    # 'spect_corr',
    # 'time_corr'
]
