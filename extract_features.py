import torch

def find_boundary_neighbors(labels):
    # Tìm các điểm có giá trị 1 hoặc 2 và có lân cận bằng 0
    padded = torch.nn.functional.pad(labels, (1, 1, 1, 1), mode='constant', value=-1)
    mask_1_or_2 = (labels == 1) | (labels == 2)
    
    # Kiểm tra các lân cận (xung quanh có giá trị 0)
    boundary_mask = ((padded[2:, 1:-1] == 0) |
                     (padded[:-2, 1:-1] == 0) |
                     (padded[1:-1, 2:] == 0) |
                     (padded[1:-1, :-2] == 0)) & mask_1_or_2
    
    return boundary_mask.nonzero(as_tuple=False)

# def find_closest_zero_with_below_1_or_2(labels, target=(200, 0)):
#     """
#     Tìm tọa độ điểm có giá trị 0 có 3 điểm lân cận theo phương dọc bên dưới đều bằng 1 hoặc 2,
#     và ít nhất một trong hai lân cận trái hoặc phải phải có giá trị 1 hoặc 2.
#     Gần nhất với tọa độ `target` (mặc định là (200, 0)).
#     """
#     # Kích thước của tensor
#     # height, width = labels.shape

#     # Tạo mask cho các điểm có 3 điểm bên dưới (y+1, y+2, y+3) là 1 hoặc 2
#     below_mask = ((labels[1:-2, :] == 1) | (labels[1:-2, :] == 2)) & \
#                 ((labels[2:-1, :] == 1) | (labels[2:-1, :] == 2)) & \
#                 ((labels[3:, :] == 1) | (labels[3:, :] == 2))

#     # Kết hợp mask để chỉ lấy các điểm 0 thỏa mãn điều kiện 3 điểm bên dưới
#     valid_zeros_mask = (labels[:-3, :] == 0) & below_mask

#     # Lấy tọa độ của các điểm 0 thỏa mãn điều kiện
#     valid_zeros_coords = valid_zeros_mask.nonzero(as_tuple=False)

#     if valid_zeros_coords.size(0) == 0:
#         return None

#     # Tạo mask cho lân cận trái và phải
#     padded_labels = torch.nn.functional.pad(labels, (1, 1), mode='constant', value=-1)

#     # Lấy các tọa độ của các điểm 0
#     y_coords = valid_zeros_coords[:, 0]
#     x_coords = valid_zeros_coords[:, 1]

#     # Kiểm tra lân cận trái và phải cho các tọa độ đã lọc
#     left_neighbor = (padded_labels[y_coords, x_coords] == 1) | (padded_labels[y_coords, x_coords] == 2)
#     right_neighbor = (padded_labels[y_coords, x_coords + 2] == 1) | (padded_labels[y_coords, x_coords + 2] == 2)

#     # Kiểm tra xem ít nhất một trong hai lân cận có giá trị 1 hoặc 2
#     neighbor_condition = left_neighbor | right_neighbor

#     # Lọc ra các tọa độ có ít nhất một điểm lân cận thỏa mãn điều kiện
#     valid_coords_with_neighbors = valid_zeros_coords[neighbor_condition]

#     if valid_coords_with_neighbors.size(0) == 0:
#         return None

#     # Tính khoảng cách Euclidean bình phương từ các điểm thỏa mãn điều kiện đến `target`
#     target_tensor = torch.tensor(target, device=labels.device).float()
#     distances_squared = torch.sum((valid_coords_with_neighbors.float() - target_tensor) ** 2, dim=1)

#     # Tìm tọa độ của điểm có khoảng cách bình phương nhỏ nhất
#     closest_zero_idx = torch.argmin(distances_squared)
#     closest_zero_coord = valid_coords_with_neighbors[closest_zero_idx]

#     return closest_zero_coord

def find_optimal_point(tensor):
    height, width = tensor.shape
    # Bước 1: Tìm các tọa độ (x, y) có giá trị 0 và giá trị từ (x;y+1) đến (x;y+3) bằng 1 hoặc 2
    condition_1 = (tensor[:, :-3] == 0) & (
        ((tensor[:, 1:-2] == 1) | (tensor[:, 1:-2] == 2)) &
        ((tensor[:, 2:-1] == 1) | (tensor[:, 2:-1] == 2)) &
        ((tensor[:, 3:] == 1) | (tensor[:, 3:] == 2))
    )

    # Tọa độ thỏa mãn điều kiện bước 1
    y_coords, x_coords = torch.nonzero(condition_1, as_tuple=True)
    
    if len(x_coords) == 0:
        return None  # Không tìm thấy điểm nào

    # Bước 2: Lọc tiếp các điểm có giá trị tại (x+1;y) hoặc (x-1;y) bằng 1 hoặc 2
    valid_points = []
    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        if (x > 0 and (tensor[y, x-1] == 1 or tensor[y, x-1] == 2)) or \
           (x < width-1 and (tensor[y, x+1] == 1 or tensor[y, x+1] == 2)):
            valid_points.append((x, y))
    
    if len(valid_points) == 0:
        return None  # Không có điểm thỏa mãn bước 2

    # Bước 3: Tìm điểm (x; y) có giá trị (300 - y)^2 + (200 - x)^2 nhỏ nhất
    min_dist = float('inf')
    best_point = None
    for x, y in valid_points:
        dist = (300 - y) ** 2 + ((200 - x)*2) ** 2
        if dist < min_dist:
            min_dist = dist
            best_point = (x, 400 - y)
    
    if best_point is not None:
        best_point = (best_point[0].item(), best_point[1].item())
    
    return best_point

def convert_to_xy(index, m):
    # Chuyển index tensor thành tọa độ (x, y) trên mặt phẳng Oxy
    y = m - 1 - index[0].item()  # Tính y từ dưới lên
    x = index[1].item()          # Tính x từ trái qua phải
    return (x, y)

def find_ABC_points(tensor):
    # Kích thước tensor là (300, 400)
    height, width = tensor.shape
    
    # Tạo mask chứa các vị trí có giá trị là 1 hoặc 2
    mask = (tensor == 1) | (tensor == 2)
    
    # Lấy các tọa độ của những điểm có giá trị 1 hoặc 2
    coords = torch.nonzero(mask, as_tuple=False)
    
    # Tìm điểm A: điểm cao nhất theo trục y
    A = coords[torch.argmin(coords[:, 0])]
    
    # Tìm điểm B: điểm gần nhất với trục Oy (tức x nhỏ nhất)
    B_candidates = coords[coords[:, 1] == torch.min(coords[:, 1])]
    B = B_candidates[torch.argmin(B_candidates[:, 0])]
    
    # Tìm điểm C: điểm xa nhất với trục Oy (tức x lớn nhất)
    C_candidates = coords[coords[:, 1] == torch.max(coords[:, 1])]
    C = C_candidates[torch.argmin(C_candidates[:, 0])]
    
    return A, B, C

def line_equation_tensor(A, B):
    # Tính hệ số a và b cho phương trình đường thẳng y = ax + b cho tensor
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    a = (B[1] - A[1]) / (B[0] - A[0] + 1e-5)  # Sử dụng epsilon để tránh chia cho 0
    b = A[1] - a * A[0]
    return a, b

def calculate_area_optimized(A, B, boundary_indices, m):
    # Tính đường thẳng AB
    a, b = line_equation_tensor(A, B)
    
    # Chuyển tọa độ điểm biên thành hệ tọa độ Oxy
    boundary_x = boundary_indices[:, 1].float()
    boundary_y = m - 1 - boundary_indices[:, 0].float()  # Chuyển về hệ trục Oxy

    # Chỉ giữ các điểm nằm trong khoảng y từ A đến B
    valid_mask = (boundary_y >= A[1]) & (boundary_y <= B[1])
    boundary_x = boundary_x[valid_mask]
    boundary_y = boundary_y[valid_mask]
    
    # Tính tọa độ x dự đoán trên đường thẳng AB cho mỗi giá trị y
    x_line = (boundary_y - b) / a
    
    # Tính diện tích bằng tổng khoảng cách giữa các x trên đường AB và các x biên
    area = torch.sum(x_line - boundary_x)
    
    return area.item()

def count_zeros_below_A(A, labels):
    # A = (x_A, y_A), trục tung là x_A
    x_A, y_A = A
    m, n = labels.shape
    
    # Cắt tensor theo trục tung x_A (tất cả các giá trị cùng cột x_A)
    column_values = labels[:, x_A]
    
    # Tìm điểm 1 đầu tiên từ dưới lên
    ones_below_A = torch.where(column_values == 1)[0]
    
    if len(ones_below_A) == 0:
        return 0  # Nếu không có ô nào có giá trị 1 dưới A, trả về 0
    
    # Lấy y_min, vị trí của ô có giá trị 1 thấp nhất
    y_min = ones_below_A.max().item()
    
    # Chỉ xét các ô giữa y_A và y_min
    if y_min <= m - 1 - y_A:
        return 0  # Không có ô nào giữa A và 1 bên dưới
    
    # Dùng slicing để đếm số ô bằng 0 giữa y_A và y_min
    zeros_in_range = column_values[m - 1 - y_A:y_min] == 0
    
    return zeros_in_range.sum().item()

def find_area_between_points_optimized(labels):
    # Tìm điểm A, B, C bằng hàm find_ABC_points mới
    A_idx, B_idx, C_idx = find_ABC_points(labels)
    
    m, n = labels.shape
    
    # Chuyển đổi từ tọa độ index sang tọa độ Oxy
    A = convert_to_xy(A_idx, m)
    B = convert_to_xy(B_idx, m)
    C = convert_to_xy(C_idx, m)
    
    # Tìm các phần tử biên
    boundary_indices = find_boundary_neighbors(labels)
    
    # Tính diện tích giữa AB và các phần tử biên
    area_AB = calculate_area_optimized(B, A, boundary_indices, m)
    area_AC = calculate_area_optimized(C, A, boundary_indices, m)

    D = find_optimal_point(labels)

    # number_zero = count_zeros_below_A(A, labels)
    
    # return area_AB, area_AC, A, B, C, number_zero
    # return A, B, C, number_zero
    return A, B, C, D, area_AB, area_AC

