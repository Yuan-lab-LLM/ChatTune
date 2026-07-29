# inpatient-住院文书

ip=$1
port=$2
if [ -z "$ip" ] || [ -z "$port" ]; then
    echo "错误：请提供 IP 地址和端口号作为参数。"
    echo "用法：$0 <IP地址> <端口号>"
    exit 1
fi

#Test 1
curl -X 'POST' \
  "http://$ip:$port/inference?request_type=inpatient&scheme=admission_record" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/inpatient-admission_record.json

  echo -e "\n==================================\n"

#Test 2
curl -X 'POST' \
  "http://$ip:$port/inference?request_type=inpatient&scheme=admission_record" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/inpatient-admission_record-modify.json

  echo -e "\n==================================\n"

#Test 3
curl -X 'POST' \
  "http://$ip:$port/inference?request_type=inpatient&scheme=admission_record" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/inpatient-admission_record-fields.json

  echo -e "\n==================================\n"

#Test 4
curl -X 'POST' \
  "http://$ip:$port/inference?request_type=inpatient&scheme=admission_record" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @data/inpatient-admission_record-prefill.json

  echo -e "\n==================================\n"
