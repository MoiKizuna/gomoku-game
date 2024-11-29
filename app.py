from flask import Flask, render_template, request, jsonify
from game_logic import GomokuGame
import asyncio
import yaml

app = Flask(__name__)


def validate_config(config):
    """验证配置文件的有效性"""
    required_fields = {
        'ai': ['think_time', 'max_depth'],
        'game': ['board_size', 'win_condition']
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"配置文件缺少 {section} 部分")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"配置文件缺少 {section}.{field} 字段")


# 加载配置
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        validate_config(config)
except Exception as e:
    print(f"加载配置文件时出错: {str(e)}")
    exit(1)

game = GomokuGame()


@app.route('/')
def index():
    board_size = config['game']['board_size']
    print(f"Debug: board_size = {board_size}")
    return render_template('index.html', board_size=board_size)


@app.route('/reset', methods=['POST'])
def reset():
    global game
    game = GomokuGame()
    return jsonify({'status': 'reset'})


@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')

    # 处理 AI 先手的特殊情况
    if x == -1 and y == -1:
        ai_move = asyncio.run(game.ai_move())
        if ai_move:
            return jsonify({'status': 'continue', 'ai_move': {'x': ai_move[0], 'y': ai_move[1]}})
        else:
            return jsonify({'status': 'error', 'message': 'AI 无法移动'}), 400

    try:
        result = game.player_move(x, y)
        if result:
            return jsonify({'status': 'win', 'winner': '玩家'})

        ai_move = asyncio.run(game.ai_move())
        if ai_move:
            if game.check_win_condition():
                return jsonify({'status': 'win', 'winner': 'AI'})
            return jsonify({'status': 'continue', 'ai_move': {'x': ai_move[0], 'y': ai_move[1]}})
        else:
            return jsonify({'status': 'error', 'message': 'AI 无法移动'}), 400

    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001)
