import cv2
import faiss
import faiss.contrib.torch_utils # added to avoid errors with gpu vs. cpu arrays
import numpy as np
import torch

def compute_camera_frustum_corners(depth_image: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Compute the 3D corners of the camera frustum in world coordinates.

    Args:
        depth_image: The depth image of shape (H, W).
        K: The camera intrinsic parameters of shape (3, 3).
        pose: The camera pose of shape (4, 4).
    Returns:
        np.ndarray: An array of 3D points representing the camera frustum corners in world coordinates.
    """
    height, width = depth_image.shape
    depth_image = depth_image[depth_image > 0]
    min_depth, max_depth = depth_image.min(), depth_image.max()

    corners = np.array(
        [
            [0, 0, min_depth], # 0 - top left near
            [width, 0, min_depth], # 1 - top right near
            [0, height, min_depth], # 2 - bottom left near
            [width, height, min_depth], # 3 - bottom right near
            [0, 0, max_depth], # 4 - top left far
            [width, 0, max_depth], # 5 - top right far
            [0, height, max_depth], # 6 - bottom left far
            [width, height, max_depth], # 7 - bottom right far
        ]
    )

    # pixel to (homogeneous) camera coordinates
    x = (corners[:, 0] - K[0, 2]) * corners[:, 2] / K[0, 0]
    y = (corners[:, 1] - K[1, 2]) * corners[:, 2] / K[1, 1]
    z = corners[:, 2]
    corners_camera = np.vstack((x, y, z, np.ones(x.shape[0]))).T

    # transform to world coordinates
    corners_world = (pose @ corners_camera.T).T[:, :3]

    return corners_world


# TODO: check this function - plane normals should point outside the frustum
def compute_camera_frustum_planes(frustum_corners: torch.Tensor) -> torch.Tensor:
    """
    Compute the six planes of the camera frustum.
    
    Args:
        frustum_corners: A tensor of camera frustum corners with shape (8, 3).
    Returns:
        torch.Tensor: A tensor of frustum planes with shape (6, 4).
    """
    # plane order: near, far, left, right, top, bottom
    # plane normals are computed as cross products of frustum edges to point outside the frustum
    planes = torch.stack(
        [
            torch.cross( # (near plane - points backwards)
                frustum_corners[2] - frustum_corners[0], # bot left near - top left near
                frustum_corners[1] - frustum_corners[0] # top right near - top left near
            ),
            torch.cross( # (far plane - points forwards)
                frustum_corners[5] - frustum_corners[4], # top right far - top left far
                frustum_corners[6] - frustum_corners[4] # bot left far - top left far                
            ),
            torch.cross( # (left plane - points to left)
                frustum_corners[4] - frustum_corners[0], # top left far - top left near
                frustum_corners[2] - frustum_corners[0] # bot left near - top left near
            ),
            torch.cross( # (right plane - points to right)
                frustum_corners[7] - frustum_corners[3], # bot right far - bot right near
                frustum_corners[1] - frustum_corners[3] # top right near - bot right near 
            ),
            torch.cross( # (top plane - points up)
                frustum_corners[5] - frustum_corners[1], # top right far - top right near
                frustum_corners[0] - frustum_corners[1] #  top left near - top right near
            ),
            torch.cross( # (bottom plane - points down)
                frustum_corners[6] - frustum_corners[2], # bot left far - bot left near
                frustum_corners[3] - frustum_corners[2] # bot right near - bot left near 
            )
        ]
    )
    plane_points = frustum_corners[
        [
            0, # for near plane, use top left near
            4, # for far plane, use top left far
            0, # for left plane, use top left near
            3, # for right plane, use bot right near
            1, # for top plane, use top right near
            2 # for bottom plane, use bot left near 
        ]
    ]
    D = torch.stack([-torch.dot(plane, plane_points[i]) for i, plane in enumerate(planes)])
    return torch.cat([planes, D[:, None]], dim=1).float()


def compute_frustum_aabb(frustum_corners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the axis-aligned bounding box (AABB) of the camera frustum.

    Args:
        frustum_corners: A tensor of camera frustum corners with shape (8, 3).
    Returns:
        torch.Tensor: The minimum and maximum corners of the AABB.
    """
    min_corner = torch.min(frustum_corners, axis=0).values
    max_corner = torch.max(frustum_corners, axis=0).values
    return min_corner, max_corner


def compute_frustum_point_ids(points: torch.Tensor, frustum_corners: torch.Tensor, device: str = "cude") -> torch.Tensor:
    """
    Compute the indices of points that are within the camera frustum.

    Args:
        points: A tensor of 3D points with shape (N, 3).
        frustum_corners: A tensor of camera frustum corners with shape (8, 3).
        device: The device to use for computation.
    Returns:
        torch.Tensor: A tensor of indices of points within the camera frustum.
    """
    if points.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    
    # move data to device
    points = points.to(device)
    frustum_corners = frustum_corners.to(device)

    # coarse check if points are within AABB of frustum
    min_corner, max_corner = compute_frustum_aabb(frustum_corners)
    inside_aabb_mask = points_inside_aabb_mask(points, min_corner, max_corner)

    # fine check if points are within frustum (TODO: High priority - check this section)
    frustum_planes = compute_camera_frustum_planes(frustum_corners)
    frustum_planes = frustum_planes.to(device)
    inside_frustum_mask = points_inside_frustum_mask(points[inside_aabb_mask], frustum_planes)

    inside_aabb_mask[inside_aabb_mask == 1] = inside_frustum_mask
    return torch.where(inside_aabb_mask)[0]

    ###########
    # import matplotlib.pyplot as plt

    # # Visualize frustum planes
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot frustum corners
    # ax.scatter(frustum_corners[:, 0], frustum_corners[:, 1], frustum_corners[:, 2], c='b', marker='o')

    # # Plot frustum planes
    # for plane in frustum_planes:
    #     xx, yy = np.meshgrid(range(10), range(10))
    #     zz = (-plane[0] * xx - plane[1] * yy - plane[3]) / plane[2]
    #     ax.plot_surface(xx, yy, zz, alpha=0.2)

    # # Set plot limits and labels
    # ax.set_xlim3d(-10, 10)
    # ax.set_ylim3d(-10, 10)
    # ax.set_zlim3d(-10, 10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # # Show the plot
    # plt.show()
    ###########   


def keyframe_optimization_sampling_distribution(num_keyframes: int, num_iterations: int, current_frame_iterations: int) -> np.ndarray:
    """
    Produces a probability distribution for selecting views based on the current iteration.
    The probability distribution is constructed by assigning current_frame_iterations to the first view 
    and uniformly distributing the remaining iterations to the rest of the views, before normalizing.

    Args:
        num_keyframes: The total number of keyframes (views).
        num_iterations: The total number of iterations planned.
        current_iteration: The current iteration number.
    Returns:
        An array representing the probability distribution over keyframes (views).
    """
    if num_keyframes == 1:
        view_distribution = np.array([1.0])
    else:
        view_distribution = np.full(num_keyframes, (num_iterations - current_frame_iterations) / (num_keyframes - 1))
        view_distribution[0] = current_frame_iterations
        view_distribution /= view_distribution.sum()
    return view_distribution


def create_point_cloud(rgb_image: np.ndarray, depth_image: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Creates a point cloud from an image, depth map, camera intrinsics, and pose.

    Args:
        rgb_image: The RGB image of shape (H, W, 3)
        depth_image: The depth map of shape (H, W)
        K: The camera intrinsic parameters of shape (3, 3)
        pose: The camera pose of shape (4, 4)
    Returns:
        A point cloud of shape (N, 6) with last dimension representing (x, y, z, r, g, b)
    """
    height, width = depth_image.shape

    # get colors for each pixel
    point_colors = rgb_image.reshape(-1, 3)

    # create mesh grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # pixel to (homogeneous) camera coordinates
    x = (u - K[0, 2]) * depth_image / K[0, 0]
    y = (v - K[1, 2]) * depth_image / K[1, 1]
    z = depth_image # TODO: check if this is correct for all RGBD data. can depth also represent distance from camera center vs. z dim in camera coordinates?
    points_camera = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    points_camera = points_camera.reshape(-1, 4)

    # camera to world coordinates
    points_world = (pose @ points_camera.T).T[:, :3]

    # plot these points in 3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2], c='b', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.savefig("pc.png")
    # print("PLOT SAVED")


    # assemble point cloud of world coordinates and colors
    point_cloud = np.concatenate((points_world, point_colors), axis=-1)
    
    return point_cloud


def geometric_edge_mask(rgb_image: np.ndarray, dilate: bool = True, RGB: bool = False) -> np.ndarray:
    """
    Compute a binary mask of geometric edges in an RGB image.

    Args:
        rgb_image (np.ndarray): An RGB image.
        dilate (bool): A boolean indicating whether to dilate the edges.
        RGB (bool): A boolean indicating whether the input image is in RGB (True) or BGR (False) format.
    Returns:
        np.ndarray: A binary mask of shape (H, W) indicating geometric edges.
    """
    assert rgb_image.dtype == np.uint8
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if RGB else cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    assert gray_image.dtype == np.uint8
    # if gray_image.dtype != np.uint8:
    #     gray_image = gray_image.astype(np.uint8)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)
    # optionally make the mask thicker
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def points_inside_aabb_mask(points: torch.Tensor, min_corner: torch.Tensor, max_corner: torch.Tensor) -> torch.Tensor:
    """
    Compute a mask of points that are inside an axis-aligned bounding box (AABB).

    Args:
        points: A tensor of 3D points with shape (N, 3).
        min_corner: The minimum corner of the AABB.
        max_corner: The maximum corner of the AABB.
    Returns:
        torch.Tensor: A boolean mask of points inside the AABB.
    """
    inside_aabb_mask = (
        (points[:, 0] >= min_corner[0])
        & (points[:, 0] <= max_corner[0])
        & (points[:, 1] >= min_corner[1])
        & (points[:, 1] <= max_corner[1])
        & (points[:, 2] >= min_corner[2])
        & (points[:, 2] <= max_corner[2])
    )

    return inside_aabb_mask


def points_inside_frustum_mask(points: torch.Tensor, frustum_planes: torch.Tensor) -> torch.Tensor:
    """
    Compute a mask of points that are inside a camera frustum.

    Args:
        points: A tensor of 3D points with shape (N, 3).
        frustum_planes: A tensor of frustum planes with shape (6, 4).
    Returns:
        torch.Tensor: A boolean mask of points inside the frustum.
    """
    num_points = points.shape[0]
    points = torch.cat([points, torch.ones(num_points, 1).to(points.device)], axis=1)
    point_distances = points @ frustum_planes.T
    inside_frustum_mask = torch.all(point_distances <= 0, axis=1)
    return inside_frustum_mask
    

def sample_pixels_based_on_gradient(rgb_image: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Sample pixels based on gradient magnitude.
    
    Args:
        rgb_image: An RGB image.
        num_samples: The number of pixels to sample.
    Returns:
        np.ndarray: An array of pixel indices.
    """
    # compute x and y image gradient magnitudes
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3) # gradient in x
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3) # gradient in y
    grad_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # generate sampling distribution from gradient magnitudes
    sampling_distribution = grad_magnitude / np.sum(grad_magnitude)
    sampling_distribution = sampling_distribution.flatten()

    # sample pixels
    sampled_indices = np.random.choice(sampling_distribution.size, size=num_samples, p=sampling_distribution)

    return sampled_indices


def select_new_point_ids(
    existing_points: torch.Tensor, 
    candidate_points: torch.Tensor,
    radius: float = 0.03,
    device: str = "cuda"
) -> torch.Tensor:
    """ 
    Select subset of the candidate points to add to the submap. Candidates are selected for addition 
    if there are no existing neighbors within the radius.

    Args:
        existing_points (frustum_points): Points of the active submap within the current view frustum of shape (N, 3)
        candidate_points: Candidate 3D Gaussian means to be added to the submap of shape (N, 3)
        radius: Radius whithin which the points are considered to be neighbors
        device: torch device to use for computation
    Returns:
        torch.Tensor: Indices of the new points that should be added to the submap of shape (N)
    """
    if existing_points.shape[0] == 0:
        return torch.arange(candidate_points.shape[0])
    
    if device == "cpu":
        points_index = faiss.IndexFlatL2(3)
    else:
        points_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))

    existing_points = existing_points.to(device)
    candidate_points = candidate_points.to(device)
    points_index.add(existing_points)

    split_pos = torch.split(candidate_points, 65535, dim=0)
    distances, ids = [], []
    for split_p in split_pos:
        distance, id = points_index.search(split_p.float(), 1)
        distances.append(distance)
        ids.append(id)
    distances = torch.cat(distances, dim=0)
    ids = torch.cat(ids, dim=0)
    neighbor_num = (distances < radius).sum(axis=1).int()
    points_index.reset()

    add_candidate_ids = torch.where(neighbor_num == 0)[0]

    return add_candidate_ids


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # load image from file
    image = cv2.imread("./data/miscellaneous_test/edge_detection_test.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = geometric_edge_mask(image, dilate=True, RGB=True)
    plt.imshow(mask, cmap="gray")
    plt.savefig("./data/miscellaneous_test/edge_detection_test_mask.jpg")
