import alibabacloud_oss_v2 as oss
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===================== OSS 配置 (统一从 .env 读取) =====================
OSS_REGION = os.getenv("OSS_REGION", "cn-beijing")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "xiaozhuteacher")


def upload_to_oss(local_file_path: str):
    """
        上传本地文件到阿里云OSS
        :param local_file_path: 本地文件路径 (支持绝对/相对路径, Windows/Linux)
        :param access_key_id: 阿里云AccessKey ID
        :param access_key_secret: 阿里云AccessKey Secret
        :return: 上传成功返回OSS文件访问地址, 失败返回None
    """
    # ===================== OSS基础配置(从 .env 读取) =====================
    region = OSS_REGION              # Bucket地域
    bucket_name = OSS_BUCKET_NAME    # Bucket名称
    oss_folder = "xiaozhuteacher/"  # 固定文件夹路径(不变)
    # 
    # 阿里云AccessKey
    ACCESS_KEY_ID = os.environ.get("Alibaba_ACCESS_KEY_ID")
    ACCESS_KEY_SECRET = os.environ.get("Alibaba_ACCESS_KEY_SECRET")

    # ===================== 路径处理 (跨系统兼容) =====================
    # 统一转换为Path对象, 自动适配Windows/Linux路径分隔符
    file_path = Path(local_file_path).resolve()  # 解析为绝对路径 (处理相对路径)

    # 校验文件是否存在
    if not file_path.exists():
        print(f" 本地文件不存在:{file_path}")
        return None
    if not file_path.is_file():
        print(f" 不是有效文件:{file_path}")
        return None

    # 提取纯文件名 (自动处理路径分隔符, 兼容所有系统)
    file_name = file_path.name
    # 拼接OSS目标路径: 固定文件夹 + 文件名
    object_name = oss_folder + file_name

    # ===================== 客户端初始化 =====================
    try:
        # 配置凭证
        credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id=ACCESS_KEY_ID,
            access_key_secret=ACCESS_KEY_SECRET
        )
        # 加载默认配置
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = region
        # 创建OSS客户端
        client = oss.Client(cfg)
        # 显式设置文件为私有 (OSS默认也是私有, 此处显式声明更稳妥), "x-oss-object-acl": "private"
        # 公共读
        headers = {
            "x-oss-object-acl": "public-read"
        }
        # ===================== 上传文件 (设置私有ACL) =====================
        with open(file_path, 'rb') as f:
            # 构建上传请求, 设置ACL为私有(private)
            request = oss.PutObjectRequest(
                bucket=bucket_name,
                key=object_name,
                body=f,
                headers=headers
            )
            result = client.put_object(request)
            print(f"   ETag: {result.etag}")
        # 拼接私有文件的访问地址 (私有文件需签名才能访问, 此处仅返回基础地址)
        oss_url = f"https://{bucket_name}.oss-{region}.aliyuncs.com/{object_name}"
        print(f"文件上传成功!")
        return oss_url
    except Exception as e:
        print(f"文件上传失败:{str(e)}")
        return None


def upload_pdf_to_oss(local_file_path: str):
    """
    上传本地 PDF 文件到阿里云 OSS (独立文件夹 pdf_documents/)
    与 upload_to_oss 逻辑完全一致, 仅 oss_folder 不同.
    :param local_file_path: 本地 PDF 文件路径
    :return: 上传成功返回 OSS 文件访问地址, 失败返回 None
    """
    region = OSS_REGION
    bucket_name = OSS_BUCKET_NAME
    oss_folder = "pdf_documents/"       # PDF 专用文件夹, 与图片文件夹隔离
    ACCESS_KEY_ID = os.environ.get("Alibaba_ACCESS_KEY_ID")
    ACCESS_KEY_SECRET = os.environ.get("Alibaba_ACCESS_KEY_SECRET")

    file_path = Path(local_file_path).resolve()
    if not file_path.exists():
        print(f" 本地文件不存在:{file_path}")
        return None
    if not file_path.is_file():
        print(f" 不是有效文件:{file_path}")
        return None

    file_name = file_path.name
    object_name = oss_folder + file_name

    try:
        credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id=ACCESS_KEY_ID,
            access_key_secret=ACCESS_KEY_SECRET
        )
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = region
        client = oss.Client(cfg)

        headers = {"x-oss-object-acl": "public-read"}
        with open(file_path, 'rb') as f:
            request = oss.PutObjectRequest(
                bucket=bucket_name,
                key=object_name,
                body=f,
                headers=headers
            )
            result = client.put_object(request)
            print(f"   ETag: {result.etag}")
        oss_url = f"https://{bucket_name}.oss-{region}.aliyuncs.com/{object_name}"
        print(f"PDF 文件上传成功!")
        return oss_url
    except Exception as e:
        print(f"PDF 文件上传失败:{str(e)}")
        return None


def upload_txt_to_oss(local_file_path: str):
    """
    上传本地 .txt 文件到阿里云 OSS (独立文件夹 medical_texts/)
    与 upload_pdf_to_oss 逻辑完全一致, 仅 oss_folder 不同.
    :param local_file_path: 本地 .txt 文件路径
    :return: 上传成功返回 OSS 文件访问地址, 失败返回 None
    """
    region = OSS_REGION
    bucket_name = OSS_BUCKET_NAME
    oss_folder = "medical_texts/"       # txt 专用文件夹, 与 PDF 和图片文件夹隔离
    ACCESS_KEY_ID = os.environ.get("Alibaba_ACCESS_KEY_ID")
    ACCESS_KEY_SECRET = os.environ.get("Alibaba_ACCESS_KEY_SECRET")

    file_path = Path(local_file_path).resolve()
    if not file_path.exists():
        print(f" 本地文件不存在:{file_path}")
        return None
    if not file_path.is_file():
        print(f" 不是有效文件:{file_path}")
        return None

    file_name = file_path.name
    object_name = oss_folder + file_name

    try:
        credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id=ACCESS_KEY_ID,
            access_key_secret=ACCESS_KEY_SECRET
        )
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = region
        client = oss.Client(cfg)

        headers = {"x-oss-object-acl": "public-read"}
        with open(file_path, 'rb') as f:
            request = oss.PutObjectRequest(
                bucket=bucket_name,
                key=object_name,
                body=f,
                headers=headers
            )
            result = client.put_object(request)
            print(f"   ETag: {result.etag}")
        oss_url = f"https://{bucket_name}.oss-{region}.aliyuncs.com/{object_name}"
        print(f"txt 文件上传成功!")
        return oss_url
    except Exception as e:
        print(f"txt 文件上传失败:{str(e)}")
        return None


if __name__ == "__main__":
    # 1. Windows绝对路径
    # local_file = r"D:\workspace\llm\lm\files\Image_1756725717883.jpg"
    # 2. Windows相对路径
    # local_file = "files/Image_1756725717883.jpg"
    # 3. Linux绝对路径
    # local_file = "/home/user/llm/files/Image_1756725717883.jpg"
    # 4. Linux相对路径
    local_file = "./files/019f3af7a3e248949508c73b3e782200_180.jpg"
    temp_file = "iform-flex-bak.zip"
    oss_url = upload_to_oss(local_file)
    print(oss_url)
    oss_url = upload_to_oss(temp_file)
    print(oss_url)