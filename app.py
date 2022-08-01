from flask import Flask, render_template, Response
from MoveNet import pose

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(MoveNet):
    while True:
        frame = MoveNet.get_pose()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    # print("done")
    return Response(gen(pose()),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
