# 数据标注平台接口测试

ip=$1
port=$2
if [ -z "$ip" ] || [ -z "$port" ]; then
    echo "错误：请提供 IP 地址和端口号作为参数。"
    echo "用法：$0 <IP地址> <端口号>"
    exit 1
fi

curl -X 'GET' \
  "http://$ip:$port/train/123456?item_type=case2chat" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/case2chat.json

  echo -e "\n==================================\n"