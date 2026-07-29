# v3-对话挂号

ip=$1
port=$2
if [ -z "$ip" ] || [ -z "$port" ]; then
    echo "错误：请提供 IP 地址和端口号作为参数。"
    echo "用法：$0 <IP地址> <端口号>"
    exit 1
fi

curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v3" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/register.json

echo -e "\n==================================\n"

curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v3" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/register-first.json

echo -e "\n==================================\n"

  curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v3" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/register-first-department.json

echo -e "\n==================================\n"

  curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v3" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/register-first-case1.json

  echo -e "\n==================================\n"

  curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v3" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/register-first-case2.json

  echo -e "\n==================================\n"

  curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v3" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/register-first-case3.json


  echo -e "\n==================================\n"

  curl -X 'POST' \
  "http://$ip:$port/inference?request_type=v3" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/register-first-case4.json
