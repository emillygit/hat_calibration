import numpy as np
import cv2

# Camera Calibration Results
camera_matrix = np.array([
    [3362.34936, 0.0, 894.223039],
    [0.0, 3378.56631, 723.946326],
    [0.0, 0.0, 1.0]
])
dist_coeffs = np.array([
    -0.725664344, 5.88357731e+01, -0.026493003, -0.00880559515, -1064.73595
])

# 3D real world coordinates from digitizer
object_points = np.array([
    [4.642, 58.069, 58.065],     # middle green
    [13.042, 62.321, 44.433],    # FP1
    [-6.977, 62.426, 46.489],    # FP2
    [13.205, 61.150, 34.556],    # left eye
    [-4.582, 62.336, 34.657],    # right eye
    [3.937, 63.464, 37.261],     # nose bridge
    [3.214, 67.762, 30.507],     # nose tip
    [27.752, 33.348, 25.141],    # LPA
    [-24.612, 36.440, 27.389],   # RPA
], dtype=np.float32)

# 2D image pixel coordinates from the labeled image
image_points = np.array([
    [575, 225],  # middle green
    [625, 300],  # FP1
    [525, 300],  # FP2
    [650, 375],  # left eye
    [525, 375],  # right eye
    [575, 375],  # nose bridge
    [575, 450],  # nose tip
    [700, 400],  # LPA
    [475, 375],  # RPA
], dtype=np.float32)

# Reshape for solvePnP
object_points = object_points.reshape(-1, 1, 3)
image_points = image_points.reshape(-1, 1, 2)

# Solve PnP
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
)

# Reproject points
projected_points, _ = cv2.projectPoints(object_points, rvec,tvec, camera_matrix, dist_coeffs)

# Compute reprojection errors
errors = np.linalg.norm(image_points - projected_points, axis=2)
mean_error = np.mean(errors)

# Print results
print("Reprojection errors per point (pixels):", errors.flatten())
print("Mean reprojection error: {:.2f} pixels".format(mean_error))
