"""Plotting for hymera kinetic runs.

    python -m hymera_plot timeseries RUN_DIR      conservation and current histories
    python -m hymera_plot profiles   RUN_DIR      collision coefficients vs momentum
    python -m hymera_plot phase-space RUN_DIR     (p, xi) histograms of the particles
    python -m hymera_plot fields     RUN_DIR      interpolated field maps (Output/plot_fields=1)
    python -m hymera_plot all        RUN_DIR      every plot the run's files allow

Each writes PNGs into RUN_DIR/plots (or -o DIR). Library use:

    from hymera_plot import io
    d = io.find_diagnostics("run/")
    d.timeseries["conservation"]["p_phi_mean"]
"""
