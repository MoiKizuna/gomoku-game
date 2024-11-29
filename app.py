from flask import Flask, render_template, request, jsonify
from game_logic import GomokuGame
import asyncio

app = Flask(__name__)
game = GomokuGame()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/move', methods=['POST'])
async def move():
    data = request.get_json()
    x, y = data['x'], data['y']

    # 处理AI先手的特殊标记
    if x == -1 and y == -1:
        ai_x, ai_y = await game.ai_move()
        return jsonify({'status': 'continue', 'ai_move': {'x': ai_x, 'y': ai_y}})

    result = game.player_move(x, y)
    if result:
        return jsonify({'status': 'win', 'winner': 'player'})

    ai_x, ai_y = await game.ai_move()
    if game.check_win(ai_x, ai_y):
        return jsonify({'status': 'win', 'winner': 'ai'})

    return jsonify({'status': 'continue', 'ai_move': {'x': ai_x, 'y': ai_y}})


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001)
