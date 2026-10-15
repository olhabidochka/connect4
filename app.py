import copy
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------------------------------------------------------------- Константи
ROWS, COLS = 6, 7          
EMPTY, HUMAN, AI = 0, 1, 2
WIN_SCORE = 10000
DEFAULT_DEPTH = 5       
BASE_DIR = Path(__file__).resolve().parent

COLUMN_ORDER = sorted(range(COLS), key=lambda c: abs(c - COLS // 2))


# --------------------------------------------------------- Ігрове поле / ходи
def create_board():
    return [[EMPTY] * COLS for _ in range(ROWS)]


def is_valid_move(board, column):
    return 0 <= column < COLS and board[0][column] == EMPTY


def get_valid_moves(board):
    return [c for c in COLUMN_ORDER if is_valid_move(board, c)]


def get_next_open_row(board, column):
    for row in range(ROWS - 1, -1, -1):
        if board[row][column] == EMPTY:
            return row
    return None


def drop_piece(board, row, column, piece):
    board[row][column] = piece


# ------------------------------------------------------------ Перевірка кінця
def find_winning_line(board, piece):
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in directions:
                cells = [(r + dr * i, c + dc * i) for i in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS and board[rr][cc] == piece
                       for rr, cc in cells):
                    return cells
    return None


def check_win(board, piece):
    return find_winning_line(board, piece) is not None


def is_draw(board):
    return not get_valid_moves(board)


def is_game_on(board):
    return not (check_win(board, HUMAN) or check_win(board, AI) or is_draw(board))


# ------------------------------------------------------------- Оцінка позиції
def evaluate_window(window, piece):
    opponent = HUMAN if piece == AI else AI
    score = 0
    mine = window.count(piece)
    theirs = window.count(opponent)
    empty = window.count(EMPTY)

    if mine == 4:
        score += WIN_SCORE
    elif mine == 3 and empty == 1:
        score += 100
    elif mine == 2 and empty == 2:
        score += 10

    if theirs == 4:
        score -= WIN_SCORE
    elif theirs == 3 and empty == 1:
        score -= 100
    elif theirs == 2 and empty == 2:
        score -= 10
    return score


def evaluate_position(board, piece=AI):
    score = 0
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += evaluate_window([board[r][c + i] for i in range(4)], piece)

    for c in range(COLS):
        for r in range(ROWS - 3):
            score += evaluate_window([board[r + i][c] for i in range(4)], piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += evaluate_window([board[r + i][c + i] for i in range(4)], piece)

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            score += evaluate_window([board[r - i][c + i] for i in range(4)], piece)
    return score


# ------------------------------------------------------------------ Minimax
def minimax(board, depth, alpha, beta, maximizing, stats=None):
    if stats is not None:
        stats["nodes"] += 1


    if check_win(board, AI):
        return None, WIN_SCORE + depth
    if check_win(board, HUMAN):
        return None, -WIN_SCORE - depth
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return None, 0
    if depth == 0:
        return None, evaluate_position(board, AI)

    if maximizing:                 
        best_score, best_col = -math.inf, valid_moves[0]
        for col in valid_moves:
            new_board = copy.deepcopy(board)
            drop_piece(new_board, get_next_open_row(new_board, col), col, AI)
            _, score = minimax(new_board, depth - 1, alpha, beta, False, stats)
            if score > best_score:
                best_score, best_col = score, col
            alpha = max(alpha, best_score)

            if alpha >= beta:       
                if stats is not None:
                    stats["prunes"] += 1
                break
        return best_col, best_score

    best_score, best_col = math.inf, valid_moves[0] 
    for col in valid_moves:
        new_board = copy.deepcopy(board)
        drop_piece(new_board, get_next_open_row(new_board, col), col, HUMAN)
        _, score = minimax(new_board, depth - 1, alpha, beta, True, stats)
        if score < best_score:
            best_score, best_col = score, col
        beta = min(beta, best_score)

        if alpha >= beta:             
            if stats is not None:
                stats["prunes"] += 1
            break
    return best_col, best_score


def get_ai_move(board, depth=DEFAULT_DEPTH, stats=None):
    column, _ = minimax(board, depth, -math.inf, math.inf, True, stats)
    return column


# ------------------------------------------------------------- Стан гри / API
class Game:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = create_board()
        self.current_player = HUMAN      
        self.valid_moves = get_valid_moves(self.board)
        self.last_move = None
        self.game_over = False
        self.winner = None

    def _place(self, column, piece):
        row = get_next_open_row(self.board, column)
        drop_piece(self.board, row, column, piece)
        self.last_move = {"player": piece, "row": row, "col": column}
        return self.last_move

    def _update_status(self, piece):
        if check_win(self.board, piece):
            self.game_over, self.winner = True, piece
        elif is_draw(self.board):
            self.game_over, self.winner = True, None
        else:
            self.current_player = HUMAN if piece == AI else AI
        self.valid_moves = [] if self.game_over else get_valid_moves(self.board)

    def play_turn(self, column, depth):
        moves, stats = [], {"nodes": 0, "prunes": 0}
        if self.game_over or not is_valid_move(self.board, column):
            return moves, stats

        moves.append(self._place(column, HUMAN))
        self._update_status(HUMAN)

        if not self.game_over:
            ai_col = get_ai_move(copy.deepcopy(self.board), depth, stats)
            moves.append(self._place(ai_col, AI))
            self._update_status(AI)
        return moves, stats

    def snapshot(self, moves=None, stats=None):
        line = None
        if self.winner:
            line = find_winning_line(self.board, self.winner)
        return {
            "board": self.board,
            "current_player": self.current_player,
            "valid_moves": self.valid_moves,
            "last_move": self.last_move,
            "game_over": self.game_over,
            "winner": self.winner,
            "win_line": line,
            "moves": moves or [],
            "stats": stats or {"nodes": 0, "prunes": 0},
        }


game = Game()
lock = threading.Lock()


# --------------------------------------------------------------- HTTP-сервер
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code=200):
        self._send(code, json.dumps(data).encode("utf-8"), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/state":
            with lock:
                return self._json(game.snapshot())
        if path in STATIC:
            filename, ctype = STATIC[path]
            return self._send(200, (BASE_DIR / filename).read_bytes(), ctype)
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)

        if self.path == "/api/new":
            with lock:
                game.reset()
                return self._json(game.snapshot())

        if self.path == "/api/move":
            try:
                column = int(payload.get("column"))
            except (TypeError, ValueError):
                return self._json({"error": "column must be int"}, 400)
            depth = max(4, min(5, int(payload.get("depth", DEFAULT_DEPTH))))
            with lock:
                if game.game_over or not is_valid_move(game.board, column):
                    return self._json({"error": "invalid move"}, 400)
                moves, stats = game.play_turn(column, depth)
                return self._json(game.snapshot(moves, stats))

        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def log_message(self, *args):   
        pass


def play_game(port=8000):
    """Запускає гру: піднімає сервер і відкриває браузер."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Connect 4 запущено: {url}  (Ctrl+C — вихід)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nГру зупинено.")
    finally:
        server.server_close()


if __name__ == "__main__":
    play_game()