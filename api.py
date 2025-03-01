from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import os

# 初始化应用
app = FastAPI()

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS配置
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 模型加载 ---
model_path = os.path.join(os.path.dirname(__file__), "football_model.pkl")

try:
    model = joblib.load(model_path)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {str(e)}")
    model = None

# --- 请求体定义 ---
class PredictionRequest(BaseModel):
    home_avg_goals: float
    away_avg_goals: float
    home_win_rate: float

# --- 前端页面 ---
@app.get("/")
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- 预测接口 ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        return {"error": "模型未正确加载"}
    
    try:
        input_data = [[
            request.home_avg_goals,
            request.away_avg_goals,
            request.home_win_rate
        ]]
        prediction = model.predict(input_data)[0]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}