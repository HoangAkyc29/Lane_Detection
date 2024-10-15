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

def convert_to_xy(index, m):
    # Chuyển index tensor thành tọa độ (x, y) trên mặt phẳng Oxy
    y = m - 1 - index[0].item()  # Tính y từ dưới lên
    x = index[1].item()          # Tính x từ trái qua phải
    return (x, y)

def find_ABC(labels):
    m, n = labels.shape
    boundary_indices = find_boundary_neighbors(labels)
    
    # Tìm điểm A (y cao nhất, tức index hàng nhỏ nhất)
    A = boundary_indices[boundary_indices[:, 0].argmin()]
    
    # Tìm điểm B (x nhỏ nhất, nếu bằng nhau thì y cao nhất)
    B_candidates = boundary_indices[boundary_indices[:, 1] == boundary_indices[:, 1].min()]
    B = B_candidates[B_candidates[:, 0].argmin()]
    
    # Tìm điểm C (x lớn nhất, nếu bằng nhau thì y cao nhất)
    C_candidates = boundary_indices[boundary_indices[:, 1] == boundary_indices[:, 1].max()]
    C = C_candidates[C_candidates[:, 0].argmin()]
    
    return convert_to_xy(A, m), convert_to_xy(B, m), convert_to_xy(C, m)

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

def find_area_between_points_optimized(labels):
    # Tìm điểm A, B, C
    A, B, C = find_ABC(labels)
    
    # Tìm các phần tử biên
    boundary_indices = find_boundary_neighbors(labels)
    
    # Tính diện tích giữa AB và các phần tử biên
    m, _ = labels.shape
    area_AB = calculate_area_optimized(B, A, boundary_indices, m)
    area_AC = calculate_area_optimized(C, A, boundary_indices, m)
    
    return area_AB, area_AC, A, B, C
