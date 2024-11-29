# 五子棋游戏

这是一个基于Flask和Python的五子棋游戏项目，支持玩家与AI对战。AI使用了迭代加深和Alpha-Beta剪枝算法来进行决策。

## 功能

- 玩家可以选择先手或后手。
- AI使用启发式评估函数进行决策。
- 支持基本的胜负判断。

## 目录结构

- `game_logic.py`：包含游戏的核心逻辑，包括AI的决策算法。
- `app.py`：Flask应用的入口，处理HTTP请求。
- `templates/index.html`：游戏的前端界面，使用HTML和JavaScript实现。

## 安装

1. 克隆此仓库到本地：

   ```bash
   git clone https://github.com/Moikizuna/gomoku-game.git
   cd gomoku-game
   ```

2. 创建并激活虚拟环境（可选）：

   ```bash
   python -m venv venv
   source venv/bin/activate  # 在Windows上使用 `venv\Scripts\activate`
   ```

3. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 运行

1. 启动Flask服务器：

   ```bash
   python app.py
   ```

2. 打开浏览器并访问 `http://localhost:5001` 开始游戏。

## 使用说明

- 点击“玩家先手”或“AI先手”按钮开始游戏。
- 在棋盘上点击空白位置下棋。
- 游戏状态会在页面上方显示。

## 贡献

欢迎提交问题和请求合并。请确保在提交之前测试您的更改。

## 许可证

此项目使用MIT许可证。详情请参阅 `LICENSE` 文件。
