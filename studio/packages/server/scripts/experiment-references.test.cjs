const assert = require('node:assert/strict');
const { test } = require('node:test');
const { modelExperimentPath } = require('../../client/src/pages/DashboardPage/RunPage/ModelManager/modelExperimentPath.ts');
const { serviceExperimentReferences } = require('../../client/src/pages/DashboardPage/RunPage/InferenceServicePanel/experimentReferences.ts');

test('service instance carries the recorded deployment ID through parsing', () => {
    const [item] = serviceExperimentReferences([{ instance_id: 'service-1', model: 'model', status: 'starting',
        ports: { vllm: '8000' }, experiment: { experimentId: 'exp-deploy', attemptId: 'attempt-1' } }]);
    assert.equal(item.experimentId, 'exp-deploy');
    assert.equal(item.instanceId, 'service-1');
    assert.equal(item.port, 8000);
    assert.equal(item.status, 'starting');
});
test('legacy services remain visible without guessing an experiment from model or port', () => {
    const [item] = serviceExperimentReferences([{ instance_id: 'legacy', model: 'exp-model', ports: { VLLM_OPENAI_PORT: 8000 } }]);
    assert.equal(item.experimentId, undefined);
    assert.equal(item.instanceId, 'legacy');
});
test('malformed status metadata cannot create experiment links', () => {
    for (const input of [null, {}, 'bad', [null, {}, 3]]) assert.deepEqual(serviceExperimentReferences(input), []);
    for (const experiment of [null, [], { experimentId: 3 }, { experimentId: '../another' }]) {
        const [item] = serviceExperimentReferences([{ instance_id: 'service', experiment, ports: { VLLM_OPENAI_PORT: 'bad' } }]);
        assert.equal(item.experimentId, undefined);
        assert.equal(item.port, 0);
    }
});

test('model card resolves inventory paths including trailing slashes and identical directory names', () => {
    assert.equal(modelExperimentPath({ path: '/models/base/', name: 'Qwen' }), '/models/base/Qwen');
    assert.equal(modelExperimentPath({ path: '/', name: 'model' }), '/model');
    assert.equal(modelExperimentPath({ path: '/models/Qwen', name: 'Qwen' }), '/models/Qwen/Qwen');
    assert.equal(modelExperimentPath({ path: '/models/中文 空格', name: 'model 1' }), '/models/中文 空格/model 1');
    assert.equal(modelExperimentPath({ path: '/models' }), undefined);
    assert.equal(modelExperimentPath({ name: 'model' }), undefined);
});
