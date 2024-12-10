from functions import LocalizationVisualizer

# Example usage
# To visualize Histogram Filter with Gaussian Distribution
# viz_histogram_gaussian = LocalizationVisualizer(filter_type='histogram', distribution_type='gaussian')
# viz_histogram_gaussian.visualize()

# To visualize Histogram Filter with Uniform Distribution
viz_histogram_uniform = LocalizationVisualizer(filter_type='histogram', distribution_type='uniform')
viz_histogram_uniform.visualize()

# # To visualize Particle Filter with Gaussian Distribution
# viz_particle_gaussian = LocalizationVisualizer(filter_type='particle', distribution_type='gaussian')
# viz_particle_gaussian.visualize()

# To visualize Particle Filter with Uniform Distribution
# viz_particle_uniform = LocalizationVisualizer(filter_type='particle', distribution_type='uniform')
# viz_particle_uniform.visualize()