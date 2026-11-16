// Raw prediction -> confirmed current movement -> completed idle/exercise/idle route.
export class EpisodeCounter {
  constructor(codes, {holdMs=1000, minScore=0.7, maxGapMs=1500}={}) {
    this.codes=codes.filter(code=>code!=='idle');
    this.enabled=codes.includes('idle');
    Object.assign(this,{holdMs,minScore,maxGapMs});
    this.counts=Object.fromEntries(this.codes.map(code=>[code,0]));
    this.interrupt();
  }
  breakCandidate(reason='연속 확인 다시 시작') {
    this.candidate=null;this.since=null;this.progressMs=0;this.last=null;
    this.reason=reason;
  }
  interrupt(reason='idle 확인 대기') {
    this.current=null;this.currentScore=null;this.confirmedAt=null;
    this.active=null;this.phase='waiting';
    this.breakCandidate(reason);
  }
  reset() {
    for (const code of this.codes) this.counts[code]=0;
    this.interrupt('횟수 초기화 · idle 확인 대기');
  }
  push(code,score,time) {
    if (!Number.isFinite(time) || !Number.isFinite(score) || score<0 || score>1 ||
        (code!=='idle' && !this.codes.includes(code))) {
      this.breakCandidate('유효한 예측 없음');return null;
    }
    if (this.last!==null && time<=this.last) return null;
    if (this.last!==null && time-this.last>this.maxGapMs) this.breakCandidate('새 예측 대기 후 연속 확인 다시 시작');
    this.last=time;
    if (score<this.minScore) {
      this.candidate=null;this.since=null;this.progressMs=0;
      this.reason='70% 미만 · 현재 동작 유지, 후보 확인 초기화';
      return null;
    }
    if (this.candidate!==code) {this.candidate=code;this.since=time;}
    this.progressMs=Math.min(time-this.since,this.holdMs);
    this.reason='같은 클래스 70% 이상 · 연속 1초 확인 중';
    if (this.progressMs<this.holdMs) return null;
    if (this.current===code) {this.reason='현재 동작 유지';return null;}
    const previous=this.current;
    this.current=code;this.currentScore=score;this.confirmedAt=time;
    let completed=null;
    if (code==='idle') {
      if (this.enabled && previous===this.active && this.active!==null) {
        completed=this.active;this.counts[completed]++;
      }
      this.active=null;this.phase='ready';
    } else if (previous==='idle') {
      this.active=code;this.phase='moving';
    } else {
      // Confirmed A -> B breaks idle -> A -> idle. Brief raw B never reaches here.
      this.active=null;this.phase='waiting';
    }
    this.reason=completed?`${completed} +1`:'현재 동작 확정';
    return completed;
  }
}
