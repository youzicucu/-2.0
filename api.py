from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import os
import requests
from datetime import datetime, timedelta
import numpy as np

# 初始化应用
app = FastAPI()

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 配置CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 配置足球数据API
FOOTBALL_DATA_API = "https://api.football-data.org/v4"
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "你的API密钥")  # 从环境变量获取
HEADERS = {'X-Auth-Token': API_KEY}

# --- 模型加载 ---
model_path = os.path.join(os.path.dirname(__file__), "football_model.pkl")

try:
    model = joblib.load(model_path)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {str(e)}")
    model = None

# --- 辅助函数 ---
def get_team_id(team_name: str) -> int:
    """获取球队ID"""
    try:
        response = requests.get(f"{FOOTBALL_DATA_API}/teams", headers=HEADERS)
        response.raise_for_status()
        
        for team in response.json()["teams"]:
            if team["name"].lower() == team_name.strip().lower():
                return team["id"]
        
        raise HTTPException(status_code=404, detail=f"未找到球队: {team_name}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API请求失败: {str(e)}")

def get_recent_matches(team_id: int, days: int = 365) -> list:
    """获取球队近期比赛"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        params = {
            "dateFrom": start_date,
            "dateTo": end_date,
            "status": "FINISHED",
            "limit": 20  # 获取最多20场比赛
        }
        
        response = requests.get(
            f"{FOOTBALL_DATA_API}/teams/{team_id}/matches",
            headers=HEADERS,
            params=params
        )
        response.raise_for_status()
        
        return response.json()["matches"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"获取比赛数据失败: {str(e)}")

def calculate_features(matches: list, is_home: bool) -> dict:
    """计算特征值"""
    valid_matches = []
    for match in matches:
        score = match.get("score", {}).get("fullTime")
        if score and None not in [score["home"], score["away"]]:
            valid_matches.append(match)
    
    if not valid_matches:
        return {"avg_goals": 0.0, "win_rate": 0.0}
    
    # 取最近5场有效比赛
    recent_matches = valid_matches[-5:]
    
    total_goals = 0
    wins = 0
    
    for match in recent_matches:
        score = match["score"]["fullTime"]
        if is_home:
            team_goals = score["home"]
            opponent_goals = score["away"]
        else:
            team_goals = score["away"]
            opponent_goals = score["home"]
        
        total_goals += team_goals
        if team_goals > opponent_goals:
            wins += 1
    
    return {
        "avg_goals": round(total_goals / len(recent_matches), 2),
        "win_rate": round(wins / len(recent_matches), 2)
    }

# --- 前端页面 ---
@app.get("/")
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- 预测接口 ---
class TeamPredictionRequest(BaseModel):
    home_team: str
    away_team: str

@app.post("/predict/teams")
async def predict_with_teams(data: TeamPredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 获取球队ID
        home_id = get_team_id(data.home_team)
        away_id = get_team_id(data.away_team)
        
        # 获取比赛数据
        home_matches = get_recent_matches(home_id)
        away_matches = get_recent_matches(away_id)
        
        # 计算特征
        home_features = calculate_features(home_matches, is_home=True)
        away_features = calculate_features(away_matches, is_home=False)
        
        # 构建模型输入
        input_data = np.array([[
            home_features["avg_goals"],
            away_features["avg_goals"],
            home_features["win_rate"]
        ]])
        
        # 进行预测
        prediction = model.predict(input_data)[0]
        
        return {
            "prediction": prediction,
            "features": {
                "home_team": data.home_team,
                "away_team": data.away_team,
                "home_avg_goals": home_features["avg_goals"],
                "away_avg_goals": away_features["avg_goals"],
                "home_win_rate": home_features["win_rate"]
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)