from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import requests
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fuzzywuzzy import fuzz
from cachetools import TTLCache
from dotenv import load_dotenv
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 初始化 FastAPI 应用
app = FastAPI()

# 配置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 配置缓存，使用 cachetools 的 TTLCache
cache = TTLCache(maxsize=100, ttl=3600)  # 缓存100个条目，TTL为1小时

# ====================
# 配置部分
# ====================
class APIConfig:
    FOOTBALL_DATA = {
        "base_url": "https://api.football-data.org/v4",
        "headers": {"X-Auth-Token": os.getenv("FOOTBALL_DATA_API_KEY")}
    }
    API_FOOTBALL = {
        "base_url": "https://v3.football.api-sports.io",
        "headers": {"x-apisports-key": os.getenv("API_FOOTBALL_KEY")}
    }

# ====================
# 数据模型部分
# ====================
class TeamPredictionRequest(BaseModel):
    home_team: str
    away_team: str

# ====================
# 核心功能部分
# ====================
# 加载模型
model_path = os.path.join(os.path.dirname(__file__), "football_model.pkl")
try:
    model = joblib.load(model_path)
    logger.info("✅ 模型加载成功")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {str(e)}")
    model = None

# 加载中文别名
def load_aliases():
    file_path = "data/team_aliases.csv"
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'gbk']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"✅ 使用 {encoding} 编码成功加载 team_aliases.csv")
            aliases_dict = {
                row['zh_name']: {
                    'id': row['id'],
                    'aliases': row['aliases'].split('、') if pd.notna(row['aliases']) else [],
                    'en_name': row['en_name']
                }
                for _, row in df.iterrows()
            }
            logger.debug(f"别名表内容: {aliases_dict}")
            return aliases_dict
        except UnicodeDecodeError as e:
            logger.error(f"❌ 使用 {encoding} 编码失败: {str(e)}")
        except FileNotFoundError:
            logger.error(f"❌ 未找到 {file_path} 文件")
            return {}
        except Exception as e:
            logger.error(f"❌ 加载 {file_path} 时发生未知错误: {str(e)}")
            return {}
    raise ValueError(f"无法以任何编码读取 {file_path}，请检查文件内容和编码")

ALIAS_MAPPING = load_aliases()

# 中文转换模块
def chinese_to_en(team_name: str) -> str:
    team_name = team_name.strip()
    logger.info(f"尝试转换球队名称: {team_name}")
    for zh_name, info in ALIAS_MAPPING.items():
        if team_name == zh_name or team_name in info['aliases']:
            logger.info(f"找到精确匹配: {team_name} -> {info['en_name']}")
            return info['en_name']
    # 模糊匹配
    best_score = 0
    best_match = None
    for zh_name, info in ALIAS_MAPPING.items():
        aliases = [zh_name] + info['aliases']
        for alias in aliases:
            score = fuzz.ratio(team_name, alias)
            if score > best_score and score > 75:
                best_score = score
                best_match = info['en_name']
    if best_match:
        logger.info(f"找到模糊匹配: {team_name} -> {best_match} (得分: {best_score})")
        return best_match
    logger.error(f"未找到匹配球队: {team_name}，请检查输入名称或别名表")
    raise ValueError(f"无法将 '{team_name}' 转换为英文名称，请检查输入或更新别名表")

# 多源球队查询
async def search_team(team_name: str) -> dict:
    try:
        en_name = chinese_to_en(team_name)
    except ValueError as e:
        raise HTTPException(404, str(e))

    cache_key = f"team:{en_name}"
    if cache_key in cache:
        logger.info(f"从缓存中获取球队信息: {en_name}")
        return cache[cache_key]

    sources = [search_football_data, search_api_football]
    for source in sources:
        try:
            result = await source(en_name)
            if result:
                logger.info(f"从 {source.__name__} 获取球队信息成功: {en_name}")
                cache[cache_key] = result
                return result
        except Exception as e:
            logger.error(f"从 {source.__name__} 获取球队信息失败: {str(e)}")
            continue
    raise HTTPException(404, f"未找到球队: {team_name}")

async def search_football_data(name: str):
    url = f"{APIConfig.FOOTBALL_DATA['base_url']}/teams"
    response = requests.get(
        url,
        headers=APIConfig.FOOTBALL_DATA['headers'],
        params={'name': name}
    )
    if response.status_code == 200:
        teams = response.json().get('teams', [])
        if teams:
            logger.debug(f"Football Data API 返回: {teams[0]}")
            return process_football_data(teams[0])
    logger.warning(f"Football Data API 未找到球队: {name}, 状态码: {response.status_code}")
    return None

async def search_api_football(name: str):
    url = f"{APIConfig.API_FOOTBALL['base_url']}/teams"
    response = requests.get(
        url,
        headers=APIConfig.API_FOOTBALL['headers'],
        params={'search': name}
    )
    if response.status_code == 200:
        data = response.json().get('response', [])
        if data:
            logger.debug(f"API Football 返回: {data[0]['team']}")
            return process_api_football(data[0]['team'])
    logger.warning(f"API Football 未找到球队: {name}, 状态码: {response.status_code}")
    return None

def process_football_data(team: dict):
    return {
        'id': team['id'],
        'name': team['name'],
        'country': team.get('area', {}).get('name', 'Unknown'),
        'venue': team.get('venue', 'Unknown')
    }

def process_api_football(team: dict):
    return {
        'id': team['id'],
        'name': team['name'],
        'country': team.get('country', 'Unknown'),
        'venue': team.get('venue', {}).get('name', 'Unknown')
    }

# ====================
# 特征计算部分
# ====================
async def get_team_features(team_id: int, is_home: bool):
    matches = await get_recent_matches(team_id)
    valid_matches = [m for m in matches if m['status'] == 'FINISHED'][-5:]
    if not valid_matches:
        logger.warning(f"球队 ID {team_id} 无有效比赛数据，返回默认值")
        return {'avg_goals': 0.0, 'win_rate': 0.0}
    
    total_goals = 0
    wins = 0
    for match in valid_matches:
        goals = match['home_goals'] if is_home else match['away_goals']
        against = match['away_goals'] if is_home else match['home_goals']
        total_goals += goals
        if goals > against:
            wins += 1
    
    features = {
        'avg_goals': round(total_goals / len(valid_matches), 2),
        'win_rate': round(wins / len(valid_matches), 2)
    }
    logger.debug(f"球队 ID {team_id} 特征: {features}")
    return features

async def get_recent_matches(team_id: int, days=365):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        response = requests.get(
            f"{APIConfig.FOOTBALL_DATA['base_url']}/teams/{team_id}/matches",
            headers=APIConfig.FOOTBALL_DATA['headers'],
            params={'dateFrom': start_date, 'dateTo': end_date, 'status': 'FINISHED', 'limit': 10}
        )
        if response.status_code == 200:
            matches = response.json().get('matches', [])
            result = [{
                'date': m['utcDate'],
                'home_goals': m['score']['fullTime']['home'],
                'away_goals': m['score']['fullTime']['away'],
                'status': m['status']
            } for m in matches]
            logger.debug(f"球队 ID {team_id} 最近比赛数据: {result}")
            return result
        logger.warning(f"获取比赛数据失败，状态码: {response.status_code}")
        return []
    except Exception as e:
        logger.error(f"获取比赛数据失败: {str(e)}")
        return []

# ====================
# 路由部分
# ====================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/teams")
async def predict_with_teams(data: TeamPredictionRequest, response: Response):
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    
    if not model:
        raise HTTPException(500, "模型未加载")
    
    try:
        logger.info(f"收到预测请求: 主队={data.home_team}, 客队={data.away_team}")
        # 获取球队信息
        home_info = await search_team(data.home_team)
        away_info = await search_team(data.away_team)
        
        # 获取特征
        home_features = await get_team_features(home_info['id'], is_home=True)
        away_features = await get_team_features(away_info['id'], is_home=False)
        
        # 准备模型输入
        input_data = np.array([[
            home_features['avg_goals'],
            away_features['avg_goals'],
            home_features['win_rate']
        ]])
        logger.debug(f"模型输入: {input_data}")
        
        # 预测
        prediction = model.predict(input_data)[0]
        logger.info(f"预测结果: {prediction}")
        
        # 确保 prediction 是可序列化的类型
        if isinstance(prediction, np.generic):
            prediction = prediction.item()
        
        result = {
            "prediction": prediction,
            "features": {
                "home_team": home_info['name'],
                "away_team": away_info['name'],
                "home_avg_goals": home_features['avg_goals'],
                "away_avg_goals": away_features['avg_goals'],
                "home_win_rate": home_features['win_rate']
            }
        }
        logger.debug(f"返回结果: {result}")
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(500, f"预测失败: {str(e)}")

# ====================
# 运行部分
# ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
