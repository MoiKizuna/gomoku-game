import numpy as np
import asyncio
import time
import hashlib
import yaml
import logging


class GomokuGame:
    def __init__(self):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.board_size = config['game']['board_size']
        self.win_condition = config['game']['win_condition']
        self.max_depth = config['ai']['max_depth']
        self.max_moves = config['ai'].get('max_moves', 20)

        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 1 for player, -1 for AI
        self.cache = {}  # 用于存储评估结果的缓存
        self.last_player_move = None  # 用于记录最后一个玩家的落子位置

        logging.basicConfig(level=logging.DEBUG)

    def player_move(self, x, y):
        logging.debug(f"玩家移动: x={x}, y={y}")
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            logging.error(f"无效的移动坐标: x={x}, y={y}")
            raise ValueError("无效的移动坐标")

        if self.board[x, y] == 0:
            self.board[x, y] = 1
            self.last_player_move = (x, y)  # 更新最后一个玩家的落子位置
            win = self.check_win(x, y)
            logging.debug(f"移动后检查胜利: {win}")
            return win
        logging.warning(f"尝试在已占用位置移动: x={x}, y={y}")
        return False

    async def ai_move(self):
        best_score = float('-inf')
        best_move = None
        total_nodes_evaluated = 0

        depth = 0
        while True:
            logging.debug(f"开始搜索深度: {depth}")

            try:
                score, move, nodes_evaluated = await asyncio.to_thread(self.iterative_deepening, depth)
                total_nodes_evaluated += nodes_evaluated
                logging.debug(
                    f"深度: {depth}, 节点数: {nodes_evaluated}, 当前最佳分数: {score}")
                if score > best_score:
                    best_score = score
                    best_move = move
            except Exception as e:
                logging.error(f"在深度 {depth} 时发生错误: {e}")
                break

            depth += 1
            if depth >= self.max_depth:
                logging.debug("达到最大深度，停止搜索")
                break

        if best_move:
            print(f"AI chooses move at {best_move} with score {best_score}")
            self.board[best_move[0], best_move[1]] = -1
        logging.debug(f"总节点数: {total_nodes_evaluated}")
        return best_move

    def iterative_deepening(self, depth):
        best_score = float('-inf')
        best_move = None
        total_nodes_evaluated = 0
        logging.debug(f"开始搜索深度: {depth}")

        # 假设 alpha_beta 返回 (score, move, nodes)
        score, move, nodes = self.alpha_beta(
            0, float('-inf'), float('inf'), True)
        total_nodes_evaluated += nodes
        if score > best_score:
            best_score = score
            best_move = move  # 从 alpha_beta 返回的最佳移动
        logging.debug(f"深度 {depth} 完成, 分数: {score}, 节点数: {nodes}")
        logging.debug(
            f"最佳移动: {best_move}, 分数: {best_score}, 总节点数: {total_nodes_evaluated}")
        return best_score, best_move, total_nodes_evaluated

    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        best_move = None
        board_hash = self.hash_board()
        cache_key = f"{board_hash}|depth:{depth}|player:{maximizing_player}"
        if cache_key in self.cache:
            logging.debug(
                f"缓存命中: key={cache_key}, score={self.cache[cache_key]}")
            return self.cache[cache_key], best_move, 0

        if self.check_win_condition() or depth == self.max_depth:
            score = self.evaluate_board()
            self.cache[cache_key] = score
            logging.debug(f"评估棋局: depth={depth}, score={score}")
            return score, best_move, 1

        moves = self.generate_moves()
        logging.debug(f"生成移动: depth={depth}, moves={moves}")
        nodes_evaluated = 0

        # 启发式排序：优先考虑评分更高的移动
        moves.sort(key=lambda move: self.heuristic_move_score(
            move), reverse=True)

        for move in moves:
            # 执行移动
            self.board[move[0], move[1]] = -1 if maximizing_player else 1
            score, _, nodes = self.alpha_beta(
                depth + 1, alpha, beta, not maximizing_player)
            # 撤销移动
            self.board[move[0], move[1]] = 0

            if maximizing_player:
                if score > alpha:
                    alpha = score
                    best_move = move
                if alpha >= beta:
                    break
            else:
                if score < beta:
                    beta = score
                    best_move = move
                if beta <= alpha:
                    break

        return (alpha if maximizing_player else beta), best_move, len(moves)

    def heuristic_move_score(self, move):
        x, y = move
        # 简单的启发式评分：离中心越近评分越高
        center = self.board_size // 2
        score = -(abs(x - center) + abs(y - center)) * 2

        # 可以加入更多启发式策略，例如考虑周围棋子的数量
        if self.is_nearby(x, y):
            score += 10  # 增加权重

        return score

    def generate_moves(self):
        high_priority_moves = set()
        medium_priority_moves = set()
        low_priority_moves = set()

        # 获取玩家最后落子的坐标
        last_player_move = self.get_last_player_move()

        # 高优先级：玩家最后落子周围
        if last_player_move:
            x, y = last_player_move
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                        high_priority_moves.add((nx, ny))

        # 中优先级：重要位置
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0 and self.is_important_position(x, y):
                    medium_priority_moves.add((x, y))

        # 低优先级：其他空位
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0 and (x, y) not in high_priority_moves and (x, y) not in medium_priority_moves:
                    low_priority_moves.add((x, y))

        # 按优先级排序
        sorted_moves = list(high_priority_moves) + \
            list(medium_priority_moves) + list(low_priority_moves)

        # 返回前 max_moves 个移动
        return sorted_moves[:self.max_moves]

    def get_last_player_move(self):
        # 返回记录的最后一个玩家的落子位置
        return self.last_player_move

    def is_important_position(self, x, y):
        # 判断当前位置是否为重要位置，例如可能导致胜利的地方
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
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] != 0:
                    return True
        return False

    def check_win(self, x, y):
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            for step in range(1, self.win_condition):
                nx, ny = x + step * dx, y + step * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break

            for step in range(1, self.win_condition):
                nx, ny = x - step * dx, y - step * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break

            if count >= self.win_condition:
                return True

        return False

    def check_win_condition(self):
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] != 0 and self.check_win(x, y):
                    return True
        return False

    def evaluate_board(self):
        score = 0
        patterns = [
            (100000000, 'OOOOO'),  # AI五连
            (100000, 'XXXXX'),  # 玩家五连
            (60000, '_OOOO_'),  # AI活四
            (80000, '_XXXX_'),  # 玩家活四
            (20000, 'OOOO_'),  # AI冲四
            (15000, 'XXXX_'),  # 玩家冲四
            (20000, '_OOOO'),  # AI冲四
            (15000, '_XXXX'),  # 玩家冲四
            (12000, 'OOO_O'),  # AI跳四
            (10000, 'XXX_X'),  # 玩家跳四
            (5000, '_OOO__'),  # AI活三
            (4000, '_XXX__'),  # 玩家活三
            (5000, '__OOO_'),  # AI活三
            (4000, '__XXX_'),  # 玩家活三
            (2500, 'OO_X_'),  # AI跳三
            (2000, 'XX_O_'),  # 玩家跳三
            (400, '_OO__'),  # AI活二
            (300, '_XX__'),  # 玩家活二
            (400, '__OO_'),  # AI活二
            (300, '__XX_'),  # 玩家活二
            (200, 'O__O'),  # AI眠二
            (150, 'X__X'),  # 玩家眠二
        ]

        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0 and self.is_nearby(x, y):
                    for player in [-1, 1]:  # -1 for AI, 1 for player
                        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                            line = self.get_line(x, y, dx, dy)
                            for value, pattern in patterns:
                                if pattern in line:
                                    if player == -1:
                                        score += value
                                    else:
                                        score -= value * 3

        return score

    def get_line(self, x, y, dx, dy):
        line = ''
        for step in range(-4, 5):
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] == -1:
                    line += 'X'
                elif self.board[nx, ny] == 1:
                    line += 'O'
                else:
                    line += '_'
            else:
                line += '_'
        return line

    def hash_board(self):
        # 使用哈希函数生成棋盘状态的唯一标识
        return hashlib.md5(self.board.tobytes()).hexdigest()
