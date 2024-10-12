import torch

def find_coordinates(tensor_data):
    # Lấy kích thước của tensor
    rows, cols = tensor_data.shape

    # Tạo các tensor dịch theo các hướng
    shift_up = torch.zeros_like(tensor_data)
    shift_down = torch.zeros_like(tensor_data)
    shift_left = torch.zeros_like(tensor_data)
    shift_right = torch.zeros_like(tensor_data)

    # Dịch các tensor
    shift_up[1:, :] = tensor_data[:-1, :]    # Dịch lên
    shift_down[:-1, :] = tensor_data[1:, :]  # Dịch xuống
    shift_left[:, 1:] = tensor_data[:, :-1]  # Dịch trái
    shift_right[:, :-1] = tensor_data[:, 1:] # Dịch phải

    # Tìm các phần tử có ít nhất 1 lân cận là 0
    neighbor_zeros = (shift_up == 0) | (shift_down == 0) | (shift_left == 0) | (shift_right == 0)

    # Tìm các phần tử có giá trị 1 hoặc 2
    valid_elements = (tensor_data == 1) | (tensor_data == 2)

    # Kết hợp hai điều kiện: phần tử phải là 1 hoặc 2 và có lân cận là 0
    result_mask = valid_elements & neighbor_zeros

    # Lấy các tọa độ thỏa mãn điều kiện
    indices = result_mask.nonzero(as_tuple=True)
    x_coords = indices[1].tolist()  # Trục Ox (cột)
    y_coords = (rows - 1 - indices[0]).tolist()  # Trục Oy (hàng)

    # Ghép các tọa độ lại
    result = list(zip(x_coords, y_coords))

    return result
