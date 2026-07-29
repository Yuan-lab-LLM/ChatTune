# v2-对话生成预问诊

ip=$1
port=$2
if [ -z "$ip" ] || [ -z "$port" ]; then
    echo "错误：请提供 IP 地址和端口号作为参数。"
    echo "用法：$0 <IP地址> <端口号>"
    exit 1
fi

curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v2" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/basicmedicalrecord.json

  echo -e "\n==================================\n"
