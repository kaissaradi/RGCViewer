# Spec: Physics Loading Optimization

## Objective
Reduce the time it takes to "Pre-compute Physics Cache" so the progress bar finishes faster and UMAP/Population panels are ready sooner.

## Current Problem
- Physics computation (ISI, ACG, FR, and Vision metrics) is done sequentially in a background thread (`StandardPlotsWorker`).
- For 1000+ clusters, this can take several minutes as each cluster is processed one-by-one.
- Large recordings exacerbate the ACG computation time.

## Proposed Solution
- **Parallel Processing**: Utilize a `ThreadPoolExecutor` or multiple QThreads to process clusters in parallel. Since the work is CPU-bound (FFT, signal processing), we can utilize all available cores.
- **Improved Caching**: Ensure the cache is persistent (done) but also optimize the serialization format if necessary.

## Technical Constraints
- Must remain thread-safe when writing to `self.standard_plot_cache`.
- Must not freeze the UI thread.
- Progress bar must correctly reflect parallel completion.
