from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
import joblib
from flask_cors import CORS
import webbrowser
import threading
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, render_template
import joblib
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载训练好的模型
vetc_model = joblib.load('vetc_model.pkl')  # 假设这是预测VETC的模型
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # 获取请求中的JSON数据

    # 提取输入的特征值
    features = [
        data['NLR'], data['GGT'],
        data['IntratumoralArtery'], data['Necrosis'],
        data['AEF'], data['InerRadScore']
    ]


    # 预测VETC概率
    vetc_prob = vetc_model.predict_proba([features])[0][1]  # 获取高危组的概率

    # 根据截断值0.52分组
    if vetc_prob > 0.52:
        vetc_group = '高危组'
    else:
        vetc_group = '低危组'


    return jsonify({
        'vetc_prediction': vetc_group,
        # 'prognosis_prediction': final_prognosis
    })


if __name__ == '__main__':
    app.run(debug=True)

