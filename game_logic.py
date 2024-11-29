import numpy as np
import asyncio
import time
import hashlib


class GomokuGame:
    def __init__(self):
        self.board = np.zeros((15, 15), dtype=int)
        self.current_player = 1  # 1 for player, -1 for AI
        self.cache = {}  # 用于存储评估结果的缓存

    def player_move(self, x, y):
        if self.board[x, y] == 0:
            self.board[x, y] = 1
            return self.check_win(x, y)
        return False

    async def ai_move(self):
        best_score = float('-inf')
        best_move = None
        start_time = time.time()
        time_limit = 1

        depth = 1
        while time.time() - start_time < time_limit:
            score, move = await asyncio.to_thread(self.iterative_deepening, depth)
            if score > best_score:
                best_score = score
                best_move = move
            depth += 1

        if best_move:
            print(f"AI chooses move at {best_move} with score {best_score}")
            self.board[best_move[0], best_move[1]] = -1
        return best_move

    def iterative_deepening(self, max_depth):
        best_score = float('-inf')
        best_move = None
        for move in self.generate_moves():
            x, y = move
            self.board[x, y] = -1
            score = self.alpha_beta(
                0, float('-inf'), float('inf'), False, max_depth)
            self.board[x, y] = 0
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

    def alpha_beta(self, depth, alpha, beta, maximizing_player, max_depth):
        board_hash = self.hash_board()
        if board_hash in self.cache:
            return self.cache[board_hash]

        if self.check_win_condition() or depth == max_depth:
            score = self.evaluate_board()
            self.cache[board_hash] = score
            return score

        moves = self.generate_moves()

        # 启发式排序：优先考虑评分更高的移动
        moves.sort(key=lambda move: self.heuristic_move_score(
            move), reverse=True)

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                x, y = move
                self.board[x, y] = -1
                eval = self.alpha_beta(
                    depth + 1, alpha, beta, False, max_depth)
                self.board[x, y] = 0
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.cache[board_hash] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                x, y = move
                self.board[x, y] = 1
                eval = self.alpha_beta(depth + 1, alpha, beta, True, max_depth)
                self.board[x, y] = 0
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.cache[board_hash] = min_eval
            return min_eval

    def heuristic_move_score(self, move):
        x, y = move
        # 简单的启发式评分：离中心越近评分越高
        return -(abs(x - 7) + abs(y - 7))

    def generate_moves(self, max_moves=15):
        # 定义优先级列表
        high_priority_moves = set()
        medium_priority_moves = set()
        low_priority_moves = set()

        # 遍历棋盘，分类移动
        for x in range(15):
            for y in range(15):
                if self.board[x, y] != 0:
                    # 查找当前棋子的周围空位
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < 15 and 0 <= ny < 15 and self.board[nx, ny] == 0:
                                high_priority_moves.add((nx, ny))

        # 查找之前的关键位置（例如，玩家或AI有潜在胜利的地方）
        for x in range(15):
            for y in range(15):
                if self.board[x, y] == 0 and self.is_important_position(x, y):
                    medium_priority_moves.add((x, y))

        # 生成其他低优先级的移动
        for x in range(15):
            for y in range(15):
                if self.board[x, y] == 0 and (x, y) not in high_priority_moves and (x, y) not in medium_priority_moves:
                    low_priority_moves.add((x, y))

        # 将移动按优先级排序
        sorted_moves = list(high_priority_moves) + \
            list(medium_priority_moves) + list(low_priority_moves)

        # 返回前 max_moves 个移动
        return sorted_moves[:max_moves]

    def is_important_position(self, x, y):
        # 判断当前位置是否为重要位置，例如可能导致胜利的地方
        # 可以根据具体需求进一步定义
        # 这里简单示例：检查该位置是否能够形成四连
        for player in [1, -1]:
            self.board[x, y] = player
            if self.check_win(x, y):
                self.board[x, y] = 0
                return True
            self.board[x, y] = 0
        return False

    def is_nearby(self, x, y):
        # 检查(x, y)附近是否有棋子
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 15 and 0 <= ny < 15 and self.board[nx, ny] != 0:
                    return True
        return False

    def check_win(self, x, y):
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            for step in range(1, 5):
                nx, ny = x + step * dx, y + step * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and self.board[nx, ny] == player:
                    count += 1
                else:
                    break

            for step in range(1, 5):
                nx, ny = x - step * dx, y - step * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and self.board[nx, ny] == player:
                    count += 1
                else:
                    break

            if count >= 5:
                return True

        return False

    def check_win_condition(self):
        for x in range(15):
            for y in range(15):
                if self.board[x, y] != 0 and self.check_win(x, y):
                    return True
        return False

    def evaluate_board(self):
        score = 0
        patterns = [
            (10000, 'XXXXX'),  # AI五连
            (10000, 'OOOOO'),  # 玩家五连
            (5000, '_XXXX_'),  # AI活四
            (5000, '_OOOO_'),  # 玩家活四
            (1000, 'XXXX_'),  # AI冲四
            (1000, 'OOOO_'),  # 玩家冲四
            (1000, '_XXXX'),  # AI冲四
            (1000, '_OOOO'),  # 玩家冲四
            (500, 'XXX_X'),  # AI跳四
            (500, 'OOO_O'),  # 玩家跳四
            (300, '_XXX__'),  # AI活三
            (300, '_OOO__'),  # 玩家活三
            (300, '__XXX_'),  # AI活三
            (300, '__OOO_'),  # 玩家活三
            (50, 'XX_X_'),  # AI跳三
            (10, '_XX__'),  # AI活二
            (10, '_OO__'),  # 玩家活二
            (10, '__XX_'),  # AI活二
            (10, '__OO_'),  # 玩家活二
        ]

        for x in range(15):
            for y in range(15):
                if self.board[x, y] == 0 and self.is_nearby(x, y):
                    for player in [-1, 1]:  # -1 for AI, 1 for player
                        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                            line = self.get_line(x, y, dx, dy)
                            print(
                                f"Checking line at ({x}, {y}) for player {player}: {line}")
                            for value, pattern in patterns:
                                if pattern in line:
                                    if player == -1:
                                        score += value
                                        print(
                                            f"AI pattern matched: {pattern} with score {value}")
                                    else:
                                        score -= value * 1.5
                                        print(
                                            f"Player pattern matched: {pattern} with score {value * 1.5}")

        print(f"Total score: {score}")
        return score

    def get_line(self, x, y, dx, dy):
        line = ''
        for step in range(-4, 5):
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < 15 and 0 <= ny < 15:
                if self.board[nx, ny] == -1:
                    line += 'X'
                elif self.board[nx, ny] == 1:
                    line += 'O'
                else:
                    line += '_'
            else:
                line += '_'
        print(f"Line at ({x}, {y}) in direction ({dx}, {dy}): {line}")
        return line

    def hash_board(self):
        # 使用哈希函数生成棋盘状态的唯一标识
        return hashlib.md5(self.board.tobytes()).hexdigest()
