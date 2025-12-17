#!/bin/bash

echo "=== 深度学习开发环境快速启动脚本 ==="

# 询问用户需要部署的镜像版本号
echo "请输入要部署的镜像版本号 (例如: v1, v2, latest):"
echo "直接按回车键将使用默认版本 latest"
read -p "版本号 [latest]: " VERSION

# 询问是否强制更新镜像
echo ""
echo "是否强制从远程更新镜像? (y/N)"
read -p "强制更新 [N]: " FORCE_UPDATE

# 设置默认版本号
if [ -z "$VERSION" ]; then
    VERSION="latest"
    echo "使用默认版本: $VERSION"
fi

# 设置镜像名称变量
IMAGE_NAME="dnn-dev-env:$VERSION"

echo "准备部署镜像版本: $IMAGE_NAME"

# 处理强制更新选项
if [ "$FORCE_UPDATE" = "y" ] || [ "$FORCE_UPDATE" = "Y" ]; then
    echo "正在从远程拉取最新镜像..."
    docker pull $IMAGE_NAME
    if [ $? -ne 0 ]; then
        echo "错误: 无法从远程拉取镜像 $IMAGE_NAME"
        exit 1
    fi
    echo "镜像更新完成"
else
    # 检查镜像是否存在
    echo "检查镜像是否存在..."
    if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
        echo "错误: 镜像 $IMAGE_NAME 不存在"
        echo "可用的镜像版本:"
        docker images $IMAGE_NAME --format "table {{.Tag}}\t{{.CreatedAt}}\t{{.Size}}" 2>/dev/null || echo "没有找到 $IMAGE_NAME 镜像"
        echo ""
        echo "提示: 您可以选择强制更新镜像来从远程下载"
        exit 1
    fi
fi

echo "镜像检查通过，启动容器..."

# 停止并删除可能存在的容器
echo "检查并停止现有容器..."
docker stop dnn-dev-jupyter 2>/dev/null || true
docker rm dnn-dev-jupyter 2>/dev/null || true

echo "启动容器 $IMAGE_NAME..."
docker run -d -p 8888:8080 \
    -v "$(pwd):/app" \
    --name dnn-dev-jupyter \
    $IMAGE_NAME
if [ $? -eq 0 ]; then
    echo "✓ 容器启动成功！"
    echo "访问地址: http://localhost:8888"
else
    echo "✗ 容器启动失败"
fi

echo ""
echo "=== 使用说明 ==="
echo "1. 在浏览器中访问: http://localhost"
echo "2. 将notebook文件放在 / 目录中"
echo "3. 将数据文件放在 data/ 目录中"
echo "4. 将模型文件放在 models/ 目录中"
echo ""
echo "停止服务: docker-compose down 或 docker stop dnn-dev-jupyter"
