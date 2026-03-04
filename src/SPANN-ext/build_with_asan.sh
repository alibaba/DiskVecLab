#!/bin/bash

# SPTAG AddressSanitizer Build Script
# 使用方法：
#   ./build_with_asan.sh debug     # Debug 模式 + ASan
#   ./build_with_asan.sh release   # Release 模式 + ASan
#   ./build_with_asan.sh releasex  # Release 模式，不启用 ASan

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取构建类型参数
BUILD_TYPE=${1:-debug}
ENABLE_ASAN="ON"

if [[ "$BUILD_TYPE" == "debug" ]]; then
    BUILD_TYPE_CMAKE="Debug"
    BUILD_DIR="cmake-build-debug-asan"
    ENABLE_ASAN="ON"
elif [[ "$BUILD_TYPE" == "release" ]]; then
    BUILD_TYPE_CMAKE="Release"
    BUILD_DIR="cmake-build-release-asan"
    ENABLE_ASAN="ON"
elif [[ "$BUILD_TYPE" == "releasex" ]]; then
    BUILD_TYPE_CMAKE="Release"
    BUILD_DIR="cmake-build-releasex"
    ENABLE_ASAN="OFF"
else
    echo -e "${RED}错误: 无效的构建类型 '$BUILD_TYPE'${NC}"
    echo "使用方法: $0 [debug|release|releasex]"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SPTAG AddressSanitizer 构建脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "构建类型: ${YELLOW}${BUILD_TYPE_CMAKE}${NC}"
echo -e "构建目录: ${YELLOW}${BUILD_DIR}${NC}"
echo ""

# 清理旧的构建
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}清理旧的构建目录...${NC}"
    rm -rf "$BUILD_DIR"
fi

# 创建构建目录
echo -e "${GREEN}创建构建目录...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置 CMake
if [[ "$ENABLE_ASAN" == "ON" ]]; then
    echo -e "${GREEN}配置 CMake (ASan: 开启)...${NC}"
else
    echo -e "${GREEN}配置 CMake (ASan: 关闭)...${NC}"
fi
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE_CMAKE" \
      -DENABLE_ASAN=${ENABLE_ASAN} \
      ..

# 编译
echo -e "${GREEN}开始编译...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    # Linux
    NUM_CORES=$(nproc)
fi

make -j$NUM_CORES

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}编译完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "可执行文件位于: ${YELLOW}${BUILD_DIR}/${BUILD_TYPE_CMAKE}/${NC}"
echo ""
if [[ "$ENABLE_ASAN" == "ON" ]]; then
    echo -e "${YELLOW}提示：运行程序时，ASan 会自动检测内存错误${NC}"
    echo ""
    echo -e "${YELLOW}可选的环境变量配置：${NC}"
    echo -e "  ${GREEN}export ASAN_OPTIONS=detect_leaks=1${NC}              # 检测内存泄漏"
    echo -e "  ${GREEN}export ASAN_OPTIONS=halt_on_error=0${NC}             # 遇到错误继续运行"
    echo -e "  ${GREEN}export ASAN_OPTIONS=verbosity=1${NC}                 # 详细输出"
    echo -e "  ${GREEN}export ASAN_OPTIONS=log_path=asan.log${NC}           # 输出到日志文件"
    echo ""
    echo -e "${YELLOW}示例运行命令：${NC}"
    echo -e "  ${GREEN}cd $BUILD_DIR/${BUILD_TYPE_CMAKE}${NC}"
    echo -e "  ${GREEN}export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1${NC}"
    echo -e "  ${GREEN}./your_program [arguments]${NC}"
    echo ""
else
    echo -e "${YELLOW}ASan 已禁用：以 Release 模式构建，无需额外环境变量${NC}"
    echo ""
    echo -e "${YELLOW}示例运行命令：${NC}"
    echo -e "  ${GREEN}cd $BUILD_DIR/${BUILD_TYPE_CMAKE}${NC}"
    echo -e "  ${GREEN}./your_program [arguments]${NC}"
    echo ""
fi

