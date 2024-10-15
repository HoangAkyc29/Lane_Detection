import torch

# def find_coordinates(tensor_data):
#     # Lấy kích thước của tensor
#     rows, cols = tensor_data.shape

#     # Tạo các tensor dịch theo các hướng
#     shift_up = torch.zeros_like(tensor_data)
#     shift_down = torch.zeros_like(tensor_data)
#     shift_left = torch.zeros_like(tensor_data)
#     shift_right = torch.zeros_like(tensor_data)

#     # Dịch các tensor
#     shift_up[1:, :] = tensor_data[:-1, :]    # Dịch lên
#     shift_down[:-1, :] = tensor_data[1:, :]  # Dịch xuống
#     shift_left[:, 1:] = tensor_data[:, :-1]  # Dịch trái
#     shift_right[:, :-1] = tensor_data[:, 1:] # Dịch phải

#     # Tìm các phần tử có ít nhất 1 lân cận là 0
#     neighbor_zeros = (shift_up == 0) | (shift_down == 0) | (shift_left == 0) | (shift_right == 0)

#     # Tìm các phần tử có giá trị 1 hoặc 2
#     valid_elements = (tensor_data == 1) | (tensor_data == 2)

#     # Kết hợp hai điều kiện: phần tử phải là 1 hoặc 2 và có lân cận là 0
#     result_mask = valid_elements & neighbor_zeros

#     # Lấy các tọa độ thỏa mãn điều kiện
#     indices = result_mask.nonzero(as_tuple=True)
#     x_coords = indices[1].tolist()  # Trục Ox (cột)
#     y_coords = (rows - 1 - indices[0]).tolist()  # Trục Oy (hàng)

#     # Ghép các tọa độ lại
#     result = list(zip(x_coords, y_coords))

#     return result

def find_boundary_neighbors(labels):
    # Tìm các điểm có giá trị 1 hoặc 2 và có lân cận bằng 0
    padded = torch.nn.functional.pad(labels, (1, 1, 1, 1), mode='constant', value=-1)
    mask_1_or_2 = (labels == 1) | (labels == 2)
    
    # Kiểm tra các lân cận (xung quanh có giá trị 0)
    boundary_mask = ((padded[2:, 1:-1] == 0) |
                     (padded[:-2, 1:-1] == 0) |
                     (padded[1:-1, 2:] == 0) |
                     (padded[1:-1, :-2] == 0)) & mask_1_or_2
    
    # print (boundary_mask)
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

def line_equation(p1, p2):
    # Tính hệ số a và b của phương trình đường thẳng y = ax + b giữa 2 điểm p1, p2
    x1, y1 = p1
    x2, y2 = p2
    a = (y2 - y1) / (x2 - x1 + 0.0001)
    b = y1 - a * x1
    return a, b

def calculate_area(A, B, boundary_indices, labels, m):
    a, b = line_equation(A, B)
    total_area = 0.0
    
    for y in range(A[1], B[1] + 1):  # Từ A.y đến B.y
        x_line = (y - b) / a  # Tọa độ x trên đường thẳng AB tại y
        y_tensor = m - 1 - y  # Chuyển về index hàng tương ứng trong tensor
        
        # Tìm x gần nhất ở hàng này có giá trị 1 hoặc 2 mà là biên
        x_candidates = boundary_indices[boundary_indices[:, 0] == y_tensor][:, 1]
        
        if len(x_candidates) > 0:
            x_nearest = x_candidates.min().item()  # x nhỏ nhất tại y đó
            total_area += (x_line - x_nearest)  # Tính khoảng cách
    return total_area

def find_area_between_points(labels):
    # Tìm điểm A, B, C
    A, B, C = find_ABC(labels)
    
    # Tìm các phần tử biên
    boundary_indices = find_boundary_neighbors(labels)
    
    # Tính diện tích giữa AB và các phần tử biên
    m, n = labels.shape
    area_AB = calculate_area(B, A, boundary_indices, labels, m)
    area_AC = calculate_area(C, A, boundary_indices, labels, m)
    
    return area_AB, area_AC, A, B, C