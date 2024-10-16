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

    # number_zero = count_zeros_below_A(A, labels)
    
    # return area_AB, area_AC, A, B, C, number_zero
    # return A, B, C, number_zero
    return A, B, C, area_AB, area_AC

