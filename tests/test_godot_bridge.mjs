import assert from 'node:assert/strict';
import {createPoseBridge} from '../web/godot-bridge.mjs';
import {EpisodeCounter} from '../web/counter.mjs';
let now=0;
const bridge=createPoseBridge(()=>now);
const counter=new EpisodeCounter(['m01','m03','m04','m09','idle']);
const snapshot=()=>JSON.parse(bridge.getSnapshotJson());
bridge.setConnection(true);
function predict(code,score=.9) {
  const data={class_id:code,label:code,score,probabilities:[score],timestamp_ms:now};
  bridge.setPrediction(data);counter.push(code,score,now);bridge.setCounter(counter);
  data.probabilities[0]=0;
  assert.equal(snapshot().raw_prediction.probabilities[0],score);
  assert.equal(snapshot().pose?.class_id ?? null,counter.current);
  now+=250;
}
for(let i=0;i<4;i++) predict('idle');assert.equal(snapshot().pose,null);
predict('idle');assert.equal(snapshot().pose.class_id,'idle');
for(let i=0;i<4;i++) predict('m01');assert.equal(snapshot().pose.class_id,'idle');
predict('m01');assert.equal(snapshot().pose.class_id,'m01');
predict('m03');assert.equal(snapshot().pose.class_id,'m01');
for(let i=0;i<5;i++) predict('idle');
assert.equal(snapshot().counter.counts.m01,1);assert.equal(snapshot().pose.class_id,'idle');
now+=2000;assert.equal(snapshot().raw_prediction,null);assert.equal(snapshot().pose.class_id,'idle');
bridge.clearPose('tracking_lost');assert.equal(snapshot().pose.class_id,'idle');
bridge.setConnection(false);assert.equal(snapshot().pose,null);assert.equal(snapshot().counter.counts.m01,1);
counter.reset();bridge.setCounter(counter);assert.equal(snapshot().counter.counts.m01,0);
console.log('Godot bridge passed: same confirmed state and counts, separate raw predictions, stale data, reset.');
