from flask import Flask, render_template, request, jsonify
from game_logic import GomokuGame
import asyncio

app = Flask(__name__)
game = GomokuGame()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/reset', methods=['POST'])
def reset():
    global game
    game = GomokuGame()
    return jsonify({'status': 'reset'})


@app.route('/move', methods=['POST'])
async def move():
    data = request.get_json()
    x, y = data['x'], data['y']

    # 如果是 AI 先手的特殊标记，先重置游戏并让 AI 下第一步
    if x == -1 and y == -1:
        global game
        game = GomokuGame()  # 重置游戏
        ai_move = await game.ai_move()
        if ai_move:
            return jsonify({'status': 'continue', 'ai_move': {'x': ai_move[0], 'y': ai_move[1]}})
        else:
            return jsonify({'status': 'error', 'message': 'AI 无法下棋'})

    # 处理玩家下棋
    result = game.player_move(x, y)
    if result:
        return jsonify({'status': 'win', 'winner': 'player'})

    # AI 下棋
    ai_move = await game.ai_move()
    if ai_move:
        if game.check_win(ai_move[0], ai_move[1]):
            return jsonify({'status': 'win', 'winner': 'ai'})
        return jsonify({'status': 'continue', 'ai_move': {'x': ai_move[0], 'y': ai_move[1]}})
    else:
        return jsonify({'status': 'error', 'message': 'AI 无法下棋'})


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001)
