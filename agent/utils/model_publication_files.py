"""Dependency-free checks run in the source container before publishing a full model."""
from pathlib import Path, PurePosixPath
import json


def validate_model(source):
    root = Path(source)
    if not root.is_absolute() or '..' in root.parts or not root.is_dir():
        raise ValueError('模型目录不存在或路径无效')
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError('模型路径不能经过符号链接')
    if not (root / 'config.json').is_file():
        if (root / 'adapter_config.json').is_file():
            raise ValueError('此目录是 LoRA 适配器，请先合并并选择完整模型目录再发布')
        raise ValueError('模型缺少 config.json，请先完成模型导出')
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not config.get('model_type'):
        raise ValueError('模型 config.json 缺少 model_type')
    if not any((root / name).is_file() for name in
               ('tokenizer.json', 'tokenizer.model', 'vocab.json', 'vocab.txt', 'spiece.model')):
        raise ValueError('模型缺少 tokenizer 文件，请先完成模型导出')
    indices = [root / name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
               if (root / name).is_file()]
    if indices:
        for index in indices:
            weight_map = json.loads(index.read_text(encoding='utf-8')).get('weight_map')
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError('模型权重索引无效')
            for name in weight_map.values():
                if not isinstance(name, str):
                    raise ValueError('模型权重索引无效')
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                    raise ValueError('模型权重索引包含无效路径')
                weight = root / name
                if not weight.is_file() or weight.stat().st_size == 0:
                    raise ValueError(f'模型权重分片缺失或为空：{name}')
    elif not any((root / name).is_file() and (root / name).stat().st_size > 0
                 for name in ('model.safetensors', 'pytorch_model.bin')):
        raise ValueError('模型缺少完整权重，请先合并适配器或完成模型导出')
    return {'valid': True}


if __name__ == '__main__':
    import sys
    print(json.dumps(validate_model(json.load(sys.stdin)['path'])))
