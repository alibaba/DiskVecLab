#!/bin/bash

# ASan 完整检查脚本 - 检测所有内存问题
# 使用方法: ./run_asan_full_check.sh [your_program] [program_args...]

set +e  # 不要在错误时退出，让程序继续运行

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

if [ "$#" -lt 1 ]; then
    echo -e "${RED}使用方法: $0 <program> [arguments...]${NC}"
    echo -e "${YELLOW}示例: $0 ./your_program input.bin output.bin${NC}"
    exit 1
fi

PROGRAM=$1
shift  # 移除第一个参数，剩余的是程序参数
PROGRAM_ARGS="$@"

# 创建日志目录
LOG_DIR="asan_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/asan_full_report_${TIMESTAMP}.log"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ASan 完整内存检查${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "程序: ${YELLOW}${PROGRAM}${NC}"
echo -e "参数: ${YELLOW}${PROGRAM_ARGS}${NC}"
echo -e "日志: ${YELLOW}${LOG_FILE}${NC}"
echo ""

# 检查程序是否存在
if [ ! -f "$PROGRAM" ]; then
    echo -e "${RED}错误: 找不到程序 '$PROGRAM'${NC}"
    echo -e "${YELLOW}请先构建程序:${NC}"
    echo -e "  ./build_with_asan.sh debug"
    exit 1
fi

# 配置 ASan 选项
export ASAN_OPTIONS="halt_on_error=0:continue_on_error=1:log_path=${LOG_FILE}:detect_leaks=1:verbosity=1:print_stats=1:print_legend=1"

# 输出 ASan 配置
echo -e "${GREEN}ASan 配置：${NC}"
echo -e "  ${YELLOW}halt_on_error=0${NC}        # 遇到错误继续运行"
echo -e "  ${YELLOW}continue_on_error=1${NC}   # 继续执行检测所有问题"
echo -e "  ${YELLOW}detect_leaks=1${NC}        # 检测内存泄漏"
echo -e "  ${YELLOW}verbosity=1${NC}           # 详细输出"
echo -e "  ${YELLOW}log_path=${LOG_FILE}${NC}"
echo ""

echo -e "${GREEN}开始运行程序...${NC}"
echo -e "${YELLOW}注意: ASan 会让程序变慢，请耐心等待${NC}"
echo ""
echo -e "========================================" | tee -a "${LOG_FILE}"
echo -e "ASan Full Check - $(date)" | tee -a "${LOG_FILE}"
echo -e "Program: ${PROGRAM} ${PROGRAM_ARGS}" | tee -a "${LOG_FILE}"
echo -e "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 运行程序
"$PROGRAM" $PROGRAM_ARGS 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}检查完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "退出码: ${YELLOW}${EXIT_CODE}${NC}"
echo ""

# 分析日志
echo -e "${GREEN}分析 ASan 报告...${NC}"
echo ""

# 统计错误类型
if grep -q "ERROR: AddressSanitizer" "${LOG_FILE}"; then
    echo -e "${RED}发现内存错误！${NC}"
    echo ""
    
    # 统计各类错误
    echo -e "${YELLOW}错误类型统计：${NC}"
    
    HEAP_OVERFLOW=$(grep -c "heap-buffer-overflow" "${LOG_FILE}" || echo "0")
    STACK_OVERFLOW=$(grep -c "stack-buffer-overflow" "${LOG_FILE}" || echo "0")
    GLOBAL_OVERFLOW=$(grep -c "global-buffer-overflow" "${LOG_FILE}" || echo "0")
    USE_AFTER_FREE=$(grep -c "heap-use-after-free" "${LOG_FILE}" || echo "0")
    USE_AFTER_RETURN=$(grep -c "stack-use-after-return" "${LOG_FILE}" || echo "0")
    USE_AFTER_SCOPE=$(grep -c "stack-use-after-scope" "${LOG_FILE}" || echo "0")
    MEMORY_LEAK=$(grep -c "detected memory leaks" "${LOG_FILE}" || echo "0")
    
    [ "$HEAP_OVERFLOW" -gt 0 ] && echo -e "  ${RED}✗ heap-buffer-overflow:   ${HEAP_OVERFLOW}${NC}"
    [ "$STACK_OVERFLOW" -gt 0 ] && echo -e "  ${RED}✗ stack-buffer-overflow:  ${STACK_OVERFLOW}${NC}"
    [ "$GLOBAL_OVERFLOW" -gt 0 ] && echo -e "  ${RED}✗ global-buffer-overflow: ${GLOBAL_OVERFLOW}${NC}"
    [ "$USE_AFTER_FREE" -gt 0 ] && echo -e "  ${RED}✗ heap-use-after-free:    ${USE_AFTER_FREE}${NC}"
    [ "$USE_AFTER_RETURN" -gt 0 ] && echo -e "  ${RED}✗ stack-use-after-return: ${USE_AFTER_RETURN}${NC}"
    [ "$USE_AFTER_SCOPE" -gt 0 ] && echo -e "  ${RED}✗ stack-use-after-scope:  ${USE_AFTER_SCOPE}${NC}"
    [ "$MEMORY_LEAK" -gt 0 ] && echo -e "  ${YELLOW}⚠ memory leaks detected:  ${MEMORY_LEAK}${NC}"
    
    echo ""
    echo -e "${YELLOW}错误位置摘要：${NC}"
    grep -A 2 "ERROR: AddressSanitizer" "${LOG_FILE}" | grep "in " | sort | uniq -c | sort -rn | head -10
    
else
    echo -e "${GREEN}✓ 没有发现 AddressSanitizer 错误${NC}"
fi

echo ""
echo -e "${BLUE}详细报告已保存到：${NC}"
echo -e "  ${GREEN}${LOG_FILE}${NC}"
echo ""

echo -e "${YELLOW}查看完整报告：${NC}"
echo -e "  ${GREEN}cat ${LOG_FILE}${NC}"
echo ""
echo -e "${YELLOW}查看所有错误摘要：${NC}"
echo -e "  ${GREEN}grep 'ERROR: AddressSanitizer' ${LOG_FILE}${NC}"
echo ""
echo -e "${YELLOW}查看特定错误类型（例如 stack-buffer-overflow）：${NC}"
echo -e "  ${GREEN}grep -A 20 'stack-buffer-overflow' ${LOG_FILE}${NC}"
echo ""

# 如果发现错误，显示第一个错误的详细信息
if grep -q "ERROR: AddressSanitizer" "${LOG_FILE}"; then
    echo -e "${RED}=== 第一个错误详情 ===${NC}"
    grep -m 1 -A 30 "ERROR: AddressSanitizer" "${LOG_FILE}"
    echo ""
    echo -e "${YELLOW}查看完整日志以了解所有错误${NC}"
fi

exit $EXIT_CODE

