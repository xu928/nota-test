from flask import Flask
app = Flask(__name__)

# 일반적인 라우트 방식입니다.
@app.route('/test')
def test():
    return "test"

app.run(host="0.0.0.0")