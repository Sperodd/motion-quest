import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {EpisodeCounter} from '../web/counter.mjs';
import {PoseRecorder} from '../web/recorder.mjs';
import {createPoseBridge} from '../web/godot-bridge.mjs';
let now=0;
const elements=new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{textContent:'',value:0,style:{},getContext:()=>({})});
  return elements.get(id);
}
const context={EpisodeCounter,PoseRecorder,createPoseBridge:()=>createPoseBridge(()=>now),
  performance:{now:()=>now},document:{getElementById:element,addEventListener(){}},
  window:{addEventListener(){}},console};
const source=fs.readFileSync(new URL('../web/app.js',import.meta.url),'utf8').split(String.fromCharCode(10)).filter(line=>!line.startsWith('import ')).join(String.fromCharCode(10));
vm.runInNewContext(source+`
config={codes:['m01','idle'],classes:['M01 Deep Squat','IDLE Standing Rest']};
counter=new EpisodeCounter(config.codes);running=true;godotBridge.setConnection(true);
globalThis.reply=handleReply;
globalThis.loseTracking=()=>{poseMissing=true;breakConfirmation('tracking lost');};
globalThis.restoreTracking=()=>{poseMissing=false;};
`,context);
function predict(code,score=.9) {
  context.reply({data:JSON.stringify({status:'prediction',class_id:code,label:code==='idle'?'IDLE Standing Rest':'M01 Deep Squat',score,
    probabilities:code==='idle'?[1-score,score]:[score,1-score],timestamp_ms:now,inference_ms:2,progress:1})});
  now+=250;
}
for(let i=0;i<5;i++) predict('idle');assert.equal(element('label').textContent,'대기 (idle)');
for(let i=0;i<4;i++) predict('m01');assert.equal(element('label').textContent,'대기 (idle)');
predict('m01');assert.equal(element('label').textContent,'딥 스쿼트');
assert.equal(JSON.parse(context.window.PoseTCN.getSnapshotJson()).pose.class_id,'m01');
context.loseTracking();predict('idle');context.restoreTracking();
for(let i=0;i<5;i++) predict('idle');
assert.equal(element('label').textContent,'대기 (idle)');
assert.equal(JSON.parse(context.window.PoseTCN.getSnapshotJson()).counter.counts.m01,1);
element('count-reset').onclick();assert.equal(element('label').textContent,'아직 확정된 동작 없음');
assert.equal(JSON.parse(context.window.PoseTCN.getSnapshotJson()).pose,null);
console.log('App integration passed: actual reply handler, display and Godot agree, 1s confirmation, no recovery cooldown.');
