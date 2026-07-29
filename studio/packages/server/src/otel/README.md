# OpenTelemetry Proto to TypeScript Generator

This guide explains how to generate TypeScript definitions from OpenTelemetry protobuf files.

## Usage

1. Clone the OpenTelemetry proto files:

```bash
git clone https://github.com/open-telemetry/opentelemetry-proto.git opentelemetry-proto
```

2. Generate TypeScript files:

```bash
npm install protoc-gen-ts
CURRENT_PATH=$(pwd)
PROJECT_PATH="${CURRENT_PATH}/../../../../"

protoc \
    --plugin="${PROJECT_PATH}/node_modules/.bin/protoc-gen-ts"  \
    --ts_out="${CURRENT_PATH}/" \
    --ts_opt=esModuleInterop=true \
    --ts_opt=stringEnums=true \
    --proto_path=./opentelemetry-proto/ \
    opentelemetry-proto/opentelemetry/proto/common/v1/*.proto \
    opentelemetry-proto/opentelemetry/proto/resource/v1/*.proto \
    opentelemetry-proto/opentelemetry/proto/trace/v1/*.proto
```

3. Clean up:

```bash
npm uninstall protoc-gen-ts
rm -rf opentelemetry-proto
```

## Options Explained

- `--plugin`: Specifies the ts-proto plugin to use
- `--ts_out`: Output directory for generated TypeScript files
- `--ts_opt=esModuleInterop=true`: Enables ES Module interop
- `--ts_opt=stringEnums=true`: Generates string enums instead of numeric enums
- `--proto_path`: Base path for proto files

## Generated Files

The script will generate TypeScript definitions for:

- Common proto files
- Resource proto files
- Trace proto files
