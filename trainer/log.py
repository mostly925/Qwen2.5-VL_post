import time, os

def _get_log_dir() -> str:
    # 从环境变量获取日志目录，如果不存在则使用默认值
    log_dir = os.environ.get('LOG_DIR', './logs')
    # 如果日志目录不存在，则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # 返回日志目录路径，确保以/结尾
    return f'{log_dir}/' if not log_dir.endswith('/') else log_dir


def log(msg: str, log_file=None):
    # 获取当前时间并格式化为字符串
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # 如果未指定日志文件
    if not log_file:
        # 打印到控制台，格式：[时间] 消息
        print(f'[{cur_time}] {msg}')
    else:
        # 以追加模式打开日志文件
        with open(f'{_get_log_dir()}{log_file}', 'a') as f:
            # 将带时间戳的消息写入文件
            f.write(f"[{cur_time}] {msg}")
