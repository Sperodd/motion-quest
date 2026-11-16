import assert from 'node:assert/strict';
import {EpisodeCounter} from '../web/counter.mjs';
const codes=['m01','m03','m04','m09','idle'];
function fixture() {
  const c=new EpisodeCounter(codes);let t=0;
  const step=(code,score=.9)=>{const event=c.push(code,score,t);t+=250;return event;};
  const stable=(code,score=.9)=>Array.from({length:5},()=>step(code,score)).filter(Boolean);
  return {c,step,stable};
}
{
  const c=new EpisodeCounter(codes);
  for(const t of [0,250,500,750,999]) c.push('idle',.7,t);
  assert.equal(c.current,null);
  c.push('idle',.7,1000);assert.equal(c.current,'idle');
  for(const t of [1250,1500,1750,2000]) c.push('m01',.7,t);
  assert.equal(c.current,'idle');
  c.push('m01',.7,2250);assert.equal(c.current,'m01'); // Exactly 1s, no extra 3s.
  for(const t of [2500,2750,3000,3250]) c.push('idle',.7,t);
  assert.equal(c.counts.m01,0);
  assert.equal(c.push('idle',.7,3500),'m01');assert.equal(c.counts.m01,1);
  c.push('idle',.9,3750);assert.equal(c.counts.m01,1);
}
{
  const {c,step,stable}=fixture();stable('idle');
  step('m01');step('m01');step('m01',.699); // Must restart, not accumulate.
  for(let i=0;i<4;i++) step('m01');assert.equal(c.current,'idle');
  step('m01');assert.equal(c.current,'m01');
  step('m03');step('m03');step('idle',.2);
  assert.equal(c.current,'m01');assert.equal(c.active,'m01');
  assert.deepEqual(stable('idle'),['m01']);
}
{
  const {c,stable}=fixture();stable('m09');stable('idle');assert.equal(c.counts.m09,0);
  stable('m01');stable('m03');stable('idle'); // A different confirmed movement breaks route.
  assert.equal(c.counts.m01,0);assert.equal(c.counts.m03,0);
  stable('m09');stable('idle');assert.equal(c.counts.m09,1);
}
{
  const {c,stable,step}=fixture();stable('idle');stable('m04');
  step('idle');step('idle');c.breakCandidate('tracking lost');
  assert.equal(c.current,'m04');assert.equal(c.progressMs,0);
  assert.deepEqual(stable('idle'),['m04']); // No recovery cooldown.
  c.interrupt();assert.equal(c.current,null);assert.equal(c.counts.m04,1);
  stable('m04');stable('idle');assert.equal(c.counts.m04,1);
  c.reset();assert.equal(c.counts.m04,0);
}
{
  const c=new EpisodeCounter(codes);
  c.push('idle',.9,0);c.push('idle',.9,2000);assert.equal(c.current,null);
  for(const t of [2250,2500,2750,3000]) c.push('idle',.9,t);
  assert.equal(c.current,'idle');
  c.push('m01',.9,100);assert.equal(c.current,'idle');
}
{
  const c=new EpisodeCounter(['m01']);
  for(const t of [0,250,500,750,1000]) c.push('m01',.9,t);
  assert.equal(c.current,'m01');assert.equal(c.enabled,false);assert.equal(c.counts.m01,0);
}
console.log('Counter passed: exact 70%/1s boundaries, continuous timing, confirmed routes, noise, disconnect, recovery.');
