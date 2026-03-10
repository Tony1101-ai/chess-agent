import cv2
import numpy as np
import torch
import torch.nn as nn
import chess
import chess.svg


class BoardCNN(nn.Module):
    """识别棋盘朝向（执白/执黑）的CNN模型"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*100*100, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ChessCNN(nn.Module):
    """识别棋子类型的CNN模型"""
    def __init__(self, num_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*25*25, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ChessBoardRecognizer:
    """国际象棋棋盘识别器"""
    
    def __init__(self, board_model_path="boardcnn.pth", chess_model_path="chesscnn.pth"):
        """
        初始化识别器
        
        Args:
            board_model_path: 棋盘朝向识别模型路径
            chess_model_path: 棋子识别模型路径
        """
        # 加载棋盘朝向模型
        self.board_model = BoardCNN(num_classes=2)
        self.board_model.load_state_dict(torch.load(board_model_path, weights_only=True))
        self.board_model.eval()
        
        # 加载棋子识别模型
        self.chess_model = ChessCNN(num_classes=12)
        self.chess_model.load_state_dict(torch.load(chess_model_path, weights_only=True))
        self.chess_model.eval()
        
        # 标签映射
        self.label_map = {'wK':0, 'wQ':1, 'wR':2, 'wB':3, 'wN':4, 'wP':5,
                         'bK':6, 'bQ':7, 'bR':8, 'bB':9, 'bN':10, 'bP':11}
        self.inv_label_map = {v:k for k, v in self.label_map.items()}
    
    def recognize(self, image_path):
        """
        识别棋盘图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            fen: 纯棋盘布局的FEN字符串，后缀 w 或 b 表示视角
        """
        # 读取图片
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘边界
        board_img, gray_board = self._detect_board(image, gray_image)
        
        # 识别棋盘朝向
        board_type = self._recognize_board_orientation(board_img)
        
        # 提取网格线
        cells = self._extract_grid_cells(board_img, gray_board)
        
        # 提取棋盘格
        patches = self._extract_patches(board_img, cells)
        
        # 识别每个格子
        chess_position = self._recognize_pieces(patches)
        
        # 转换为FEN格式（仅棋盘布局）
        fen = self._list_to_fen(chess_position)
        
        # 如果是黑方视角，翻转棋盘并标记 b，否则标记 w
        if board_type:
            fen = self._rotate_fen(fen)
            fen = fen + " b"
        else:
            fen = fen + " w"
        
        return fen
    
    def _detect_board(self, image, gray_image):
        """检测棋盘边界"""
        edges = cv2.Canny(gray_image, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        best_area = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            ratio = w / h if h != 0 else 0
            if 0.8 < ratio < 1.25 and area > best_area:
                best_area = area
                best_rect = (x, y, w, h)
        
        x, y, w, h = best_rect
        board = image[y:y+h, x:x+w]
        gray_board = gray_image[y:y+h, x:x+w]
        
        return board, gray_board
    
    def _recognize_board_orientation(self, board_img):
        """识别棋盘朝向（0为执白，1为执黑）"""
        img = cv2.resize(board_img, (400, 400))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            board_type = self.board_model(img).argmax(dim=-1).item()
        
        return board_type
    
    def _extract_grid_cells(self, board_img, gray_board):
        """提取棋盘网格"""
        h, w = gray_board.shape
        
        # 图像处理
        blur = cv2.GaussianBlur(gray_board, (7, 7), 0)
        _, otsu_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny = cv2.Canny(otsu_binary, 80, 200)
        kernel = np.ones((3, 3), np.uint8)
        board_dilation = cv2.dilate(canny, kernel, iterations=1)
        
        # 检测直线
        lines = cv2.HoughLinesP(
            board_dilation,
            1,
            np.pi/180,
            threshold=900,
            minLineLength=450,
            maxLineGap=20
        )
        
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # 水平线
                h_lines.append((y1 + y2) // 2)
            elif abs(x1 - x2) < 10:  # 垂直线
                v_lines.append((x1 + x2) // 2)
        
        # 添加边界
        h_lines += [0, h-1]
        v_lines += [0, w-1]
        
        # 聚类
        h_lines = self._cluster_and_center(h_lines, tol=10)
        v_lines = self._cluster_and_center(v_lines, tol=10)
        
        # 生成格子
        cells = []
        for j in range(len(h_lines) - 1):
            for i in range(len(v_lines) - 1):
                x1 = v_lines[i]
                x2 = v_lines[i+1]
                y1 = h_lines[j]
                y2 = h_lines[j+1]
                cells.append((x1, y1, x2, y2))
        
        return cells
    
    def _cluster_and_center(self, vals, tol=10):
        """聚类并计算中心"""
        vals = sorted(vals)
        clusters = []
        for v in vals:
            if not clusters or abs(v - clusters[-1][-1]) > tol:
                clusters.append([v])
            else:
                clusters[-1].append(v)
        centers = [int(sum(c)/len(c)) for c in clusters]
        return centers
    
    def _extract_patches(self, board_img, cells):
        """提取每个格子的图像块"""
        patches = []
        for (x1, y1, x2, y2) in cells:
            patch = board_img[y1:y2, x1:x2]
            patches.append(patch)
        return patches
    
    def _is_empty(self, patch, std_thresh=20):
        """判断格子是否为空"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return gray.std() < std_thresh
    
    def _recognize_pieces(self, patches):
        """识别所有格子的棋子"""
        occupied_flags = []
        occupied_patches = []
        
        for p in patches:
            is_occupied = not self._is_empty(p)
            occupied_flags.append(is_occupied)
            if is_occupied:
                occupied_patches.append(p)
        
        # 识别有棋子的格子
        occupied_labels = []
        for patch in occupied_patches:
            img = self._transform_chess_img(patch)
            with torch.no_grad():
                out = self.chess_model(img).argmax(dim=-1).item()
            occupied_labels.append(self.inv_label_map[out])
        
        # 合并结果
        chess_position = []
        j = 0
        for flag in occupied_flags:
            if flag:
                chess_position.append(occupied_labels[j])
                j += 1
            else:
                chess_position.append(0)
        
        return chess_position
    
    def _transform_chess_img(self, img):
        """转换棋子图像用于模型输入"""
        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img
    
    def _list_to_fen(self, board):
        """将棋盘列表转换为FEN格式（仅棋盘布局）"""
        fen_rows = []
        for r in range(8):
            row = board[r*8:(r+1)*8]
            fen_row = ""
            empty = 0
            for cell in row:
                if cell == 0:
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    color = cell[0]  # 'w' or 'b'
                    piece = cell[1]  # 'P','K','Q','R','B','N'
                    if color == 'w':
                        fen_row += piece.upper()
                    else:
                        fen_row += piece.lower()
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        return "/".join(fen_rows)  # 只返回棋盘布局，不加 w - - 0 1
    
    def _rotate_fen(self, fen):
        """翻转FEN字符串（用于黑方视角）"""
        rows = fen.split("/")
        rows = rows[::-1]  # 上下翻转
        new_rows = []
        for r in rows:
            expanded = []
            for c in r:
                if c.isdigit():
                    expanded += ['.'] * int(c)
                else:
                    expanded.append(c)
            expanded = expanded[::-1]  # 左右翻转
            out = ""
            cnt = 0
            for x in expanded:
                if x == '.':
                    cnt += 1
                else:
                    if cnt:
                        out += str(cnt)
                        cnt = 0
                    out += x
            if cnt:
                out += str(cnt)
            new_rows.append(out)
        return "/".join(new_rows)


def recognize_chess_board(image_path, board_model_path="boardcnn.pth", chess_model_path="chesscnn.pth"):
    """
    便捷函数：识别国际象棋棋盘
    
    Args:
        image_path: 图片路径
        board_model_path: 棋盘朝向识别模型路径
        chess_model_path: 棋子识别模型路径
        
    Returns:
        fen: 纯棋盘布局的FEN字符串，后缀 w 或 b 表示视角
    """
    recognizer = ChessBoardRecognizer(board_model_path, chess_model_path)
    return recognizer.recognize(image_path)


if __name__ == "__main__":
    # 使用示例
    image_path = "test-images/test-07.jpg"
    fen = recognize_chess_board(image_path)
    print(fen)
