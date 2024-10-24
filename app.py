# encoding:utf-8

import os
import signal
import sys
import time
import threading
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from pydantic import BaseModel
import uvicorn
import json
from datetime import datetime
from typing import Optional
import io
import redis
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, HTTPException, status
from fastapi.openapi.utils import get_openapi

from channel import channel_factory
from common import const
from config import conf, load_config
from plugins import *
import logging
from channel.chat_channel import ChatChannel, Reply, ReplyType, Context, ContextType

logger = logging.getLogger(__name__)

# 在文件顶部的导入部分添加或确保有以下导入
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.openapi.utils import get_openapi

# 修改FastAPI应用的初始化
app = FastAPI(title="dify微信消息发送")


# 添加以下函数来自定义OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="dify微信消息发送",
        version="1.0.0",
        description="这是一个用于发送微信消息的API",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# 设置自定义OpenAPI schema
app.openapi = custom_openapi


# 定义请求模型
class BaseMessageRequest(BaseModel):
    user_id: str


class TextMessageRequest(BaseMessageRequest):
    message: str


class ImageUrlMessageRequest(BaseMessageRequest):
    image_url: str


class VideoUrlMessageRequest(BaseMessageRequest):
    video_url: str


# 全局变量，用于存储 WechatChannel 实例
wechat_channel = None
channel_ready = threading.Event()

# 在全局变量部分添加 Redis 客户端
# 连接到Redis服务器，包含密码认证
redis_client = redis.Redis(
    host=conf().get("redis_host", "localhost"),
    port=conf().get("redis_port", 6379),
    db=conf().get("redis_db", 0),
    password=conf().get("redis_password", "")
)

# 在全局变量部分添加以下内容
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = conf().get_api_username()
    correct_password = conf().get_api_password()
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的用户名或密码",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def sigterm_handler_wrap(_signo):
    old_handler = signal.getsignal(_signo)

    def func(_signo, _stack_frame):
        logger.info("signal {} received, exiting...".format(_signo))
        conf().save_user_datas()
        if callable(old_handler):  # check old_handler
            return old_handler(_signo, _stack_frame)
        sys.exit(0)

    signal.signal(_signo, func)


def start_channel(channel_name: str):
    global wechat_channel
    channel = channel_factory.create_channel(channel_name)
    if channel_name in ["wx", "wxy", "terminal", "wechatmp", "wechatmp_service", "wechatcom_app", "wework",
                        "wechatcom_service", "gewechat", const.FEISHU, const.DINGTALK]:
        PluginManager().load_plugins()

    if conf().get("use_linkai"):
        try:
            from common import linkai_client
            threading.Thread(target=linkai_client.start, args=(channel,)).start()
        except Exception as e:
            logger.error(f"Failed to start linkai client: {e}")

    wechat_channel = channel
    channel_ready.set()  # 设置事件表示 channel 已经准备就绪
    channel.startup()


# 修改保存 receiver ID 的函数
def save_receiver_id(receiver_id):
    try:
        # 首先读取现有的缓存
        try:
            with open('receiver_cache.json', 'r') as f:
                receiver_cache = json.load(f)
        except FileNotFoundError:
            receiver_cache = {}

        if receiver_id not in receiver_cache:
            timestamp = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
            receiver_cache[receiver_id] = timestamp
            # 将更新后的缓存保存到文件
            with open('receiver_cache.json', 'w') as f:
                json.dump(receiver_cache, f)
            logger.info(f"Saved new receiver ID: {receiver_id}")
        else:
            logger.debug(f"Receiver ID already exists: {receiver_id}")
    except Exception as e:
        logger.error(f"Error saving receiver ID: {e}")


# 修改发送文本消息的 API 端点
@app.post("/send_text_message")
async def send_text_message(request: TextMessageRequest, username: str = Depends(verify_credentials)):
    if not channel_ready.is_set():
        raise HTTPException(status_code=503, detail="WechatChannel is not ready")

    try:
        reply = Reply(ReplyType.TEXT, request.message)
        context = Context(ContextType.TEXT)
        context.content = request.message
        context.kwargs = {'receiver': request.user_id}

        wechat_channel.send(reply, context)
        return {"status": "success", "message": "Text message sent successfully"}
    except Exception as e:
        logger.error(f"Failed to send text message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send text message: {str(e)}")


# 添加发送网络图片的 API 端点
@app.post("/send_image_url")
async def send_image_url(request: ImageUrlMessageRequest, username: str = Depends(verify_credentials)):
    if not channel_ready.is_set():
        raise HTTPException(status_code=503, detail="WechatChannel is not ready")

    try:
        reply = Reply(ReplyType.IMAGE_URL, request.image_url)
        context = Context(ContextType.IMAGE)
        context.kwargs = {'receiver': request.user_id}

        wechat_channel.send(reply, context)
        return {"status": "success", "message": "Image URL sent successfully"}
    except Exception as e:
        logger.error(f"Failed to send image URL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send image URL: {str(e)}")


# 添加发送本地图片文件的 API 端点
@app.post("/send_image_file")
async def send_image_file(user_id: str, file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    if not channel_ready.is_set():
        raise HTTPException(status_code=503, detail="WechatChannel is not ready")

    try:
        image_content = await file.read()
        reply = Reply(ReplyType.IMAGE, io.BytesIO(image_content))
        context = Context(ContextType.IMAGE)
        context.kwargs = {'receiver': user_id}

        wechat_channel.send(reply, context)
        return {"status": "success", "message": "Image file sent successfully"}
    except Exception as e:
        logger.error(f"Failed to send image file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send image file: {str(e)}")


# 添加发送文件的 API 端点
@app.post("/send_file")
async def send_file(user_id: str, file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    if not channel_ready.is_set():
        raise HTTPException(status_code=503, detail="WechatChannel is not ready")

    try:
        file_content = await file.read()
        reply = Reply(ReplyType.FILE, io.BytesIO(file_content))
        context = Context(ContextType.FILE)
        context.kwargs = {'receiver': user_id}

        wechat_channel.send(reply, context)
        return {"status": "success", "message": "File sent successfully"}
    except Exception as e:
        logger.error(f"Failed to send file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send file: {str(e)}")


# 添加发送网视频的 API 端点
@app.post("/send_video_url")
async def send_video_url(request: VideoUrlMessageRequest, username: str = Depends(verify_credentials)):
    if not channel_ready.is_set():
        raise HTTPException(status_code=503, detail="WechatChannel is not ready")

    try:
        reply = Reply(ReplyType.VIDEO_URL, request.video_url)
        context = Context(ContextType.VIDEO)
        context.kwargs = {'receiver': request.user_id}

        wechat_channel.send(reply, context)
        return {"status": "success", "message": "Video URL sent successfully"}
    except Exception as e:
        logger.error(f"Failed to send video URL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send video URL: {str(e)}")


# 添加发送本地视频文件的 API 端点
@app.post("/send_video_file")
async def send_video_file(user_id: str, file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    if not channel_ready.is_set():
        raise HTTPException(status_code=503, detail="WechatChannel is not ready")

    try:
        video_content = await file.read()
        reply = Reply(ReplyType.VIDEO, io.BytesIO(video_content))
        context = Context(ContextType.VIDEO)
        context.kwargs = {'receiver': user_id}

        wechat_channel.send(reply, context)
        return {"status": "success", "message": "Video file sent successfully"}
    except Exception as e:
        logger.error(f"Failed to send video file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send video file: {str(e)}")


# 修改获取 receiver IDs 的 API 端点
@app.get("/get_receivers")
async def get_receivers(username: str = Depends(verify_credentials)):
    try:
        with open('receiver_cache.json', 'r') as f:
            receiver_cache = json.load(f)
        return receiver_cache
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error(f"Error reading receiver cache: {e}")
        raise HTTPException(status_code=500, detail="Error reading receiver cache")


def clear_receiver_cache():
    """清空 receiver_cache.json 文件"""
    try:
        with open('receiver_cache.json', 'w') as f:
            json.dump({}, f)
        logger.info("Cleared receiver_cache.json")
    except Exception as e:
        logger.error(f"Error clearing receiver_cache.json: {e}")


# 添加新的请模型用于 Redis 操作
class RedisWriteRequest(BaseModel):
    key: str
    value: str
    expire: Optional[int] = None  # 过期时间（秒），可选参数


class RedisReadRequest(BaseModel):
    key: str


# 添加向 Redis 写入数据的 API 端点
@app.post("/write_to_redis")
async def write_to_redis(request: RedisWriteRequest, username: str = Depends(verify_credentials)):
    try:
        if request.expire is not None:
            # 如果提供了过期时间，使用 setex 命令
            redis_client.setex(request.key, request.expire, request.value)
        else:
            # 如果没有提供过期时间，使用普通的 set 命令
            redis_client.set(request.key, request.value)

        return {
            "status": "success",
            "message": f"Data written to Redis with key: {request.key}",
            "expire": request.expire if request.expire is not None else "No expiration set"
        }
    except Exception as e:
        logger.error(f"Failed to write to Redis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to write to Redis: {str(e)}")


# 添加从 Redis 读取数据的 API 端点
@app.post("/read_from_redis")
async def read_from_redis(request: RedisReadRequest, username: str = Depends(verify_credentials)):
    try:
        value = redis_client.get(request.key)
        if value is None:
            return {"status": "not_found", "message": f"No data found for key: {request.key}"}
        return {"status": "success", "key": request.key, "value": value.decode('utf-8')}
    except Exception as e:
        logger.error(f"Failed to read from Redis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read from Redis: {str(e)}")


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8080)


def run():
    global receiver_cache
    try:
        # 在加载配置之前清空 receiver_cache.json
        clear_receiver_cache()

        # load config
        load_config()
        # ctrl + c
        sigterm_handler_wrap(signal.SIGINT)
        # kill signal
        sigterm_handler_wrap(signal.SIGTERM)

        # create channel
        channel_name = conf().get("channel_type", "wx")

        if "--cmd" in sys.argv:
            channel_name = "terminal"

        if channel_name == "wxy":
            os.environ["WECHATY_LOG"] = "warn"

        # 创建并启动 FastAPI 线程
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()

        # 启动 WechatChannel
        start_channel(channel_name)

        # 主线程等待
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error("App startup failed!")
        logger.exception(e)


if __name__ == "__main__":
    run()
